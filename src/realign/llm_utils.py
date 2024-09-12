import os
import json
import asyncio
from dataclasses import dataclass, fields
from typing import Any, Optional, Callable, Union, Coroutine
import hashlib
import yaml
import time


from jinja2 import Template

from litellm import ModelResponse, aembedding, acompletion, validate_environment
from litellm.types.utils import Choices
import litellm

from realign.router import Router
from realign.utils import bcolors, run_async, dotdict
from realign.evaluators import evaluator, aevaluator


# this flag helps litellm modify params to ensure that model-specific requirements are met
litellm.modify_params = True

# initialize the global request router
router = Router()
        
# Register the cleanup function
# atexit.register(lambda: router or router.__del__())

@dataclass
class OpenAIMessage:
    role: str
    content: str | dict[str, str]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def __getitem__(self, key: str):
        if key == 'role':
            return self.role
        elif key == 'content':
            return self.content
        elif key == 'tool_call_id':
            return self.tool_call_id
        elif key == 'name':
            return self.name
        else:
            raise KeyError(key)

    def __dict__(self):
        if self.name is None and self.tool_call_id is None:
            return {
                "role": str(self.role),
                "content": str(self.content)
            }
        else:
            return {
                "role": str(self.role),
                "content": str(self.content),
                "name": str(self.name),
                "tool_call_id": str(self.tool_call_id)
            }

    def __eq__(self, other):
        if isinstance(other, dict):
            other = OpenAIMessage(**other)
        elif not isinstance(other, OpenAIMessage):
            return False
        
        return self.role == other.role and self.content == other.content


@dataclass
class RunData:
    final_state: Any
    run_id: Optional[int] = None

    def __dict__(self):
        return {"run_id": self.run_id, "final_state": self.final_state}

    def __repr__(self) -> str:
        return str(self.__dict__())

    def compute_hash(self, hash_algorithm="sha256"):
        """
        Compute a hash of a RunData.

        :param obj: The object to hash
        :param hash_algorithm: The hash algorithm to use (default is 'sha256')
        :return: A hexadecimal string representation of the hash
        """
        # Convert the object to a JSON string
        json_string = json.dumps(self.__dict__(), sort_keys=True, default=str)

        # Create a hash object with the specified algorithm
        hash_object = hashlib.new(hash_algorithm)

        # Update the hash object with the JSON string (encoded to bytes)
        hash_object.update(json_string.encode("utf-8"))

        # Return the hexadecimal representation of the hash
        return hash_object.hexdigest()


@dataclass
class State:
    messages: list[OpenAIMessage]

    def __init__(self):
        self.messages = []

    def __repr__(self) -> str:
        return str_msgs(self.messages[1:])


@dataclass
class AgentSettings:
    # litellm model name. Refer to https://docs.litellm.ai/docs/providers.
    model: str

    # API key env variable name.
    # If not provided, defaults to <MODEL_PROVIDER>_API_KEY format
    api_key: Optional[str] = None

    # hyperparam dictionary in OpenAI format, eg. { 'temperature': 0.8 }
    hyperparams: Optional[dict[str, Any]] = None

    # literal system prompt
    # if provided, template and template_params will be ignored
    system_prompt: Optional[str] = None

    # Jinja template and prompt_param dictionary to render it
    # string key for the template. Actual templates defined in realign.prompts
    template_params: Optional[dict[str, str]] = None
    template: Optional[str] = None

    # json_mode for the response format
    json_mode: Optional[bool] = False
    
    # tools
    tools: Optional[list[dict | str]] = None

    # user or assistant
    role: str = "assistant"
    
    # initialize the messages
    # system messages will be ignored, since they must be defined in the template / system_prompt
    init_messages: Optional[list[OpenAIMessage]] = None

    def resolve_response_format(self) -> str:
        if self.json_mode:
            return {"type": "json_object"}
        return None

    def resolve_system_prompt(self) -> str | None:
        prompt_to_render = ""
        system_prompt = self.system_prompt
        template = self.template
        if system_prompt is None:
            if template is None:
                return None
            else:
                prompt_to_render = self.resolve_prompt_template(template)
        else:
            prompt_to_render = system_prompt

        jinja_template = Template(prompt_to_render)
        template_params = self.template_params

        if template_params is None:
            return jinja_template.render({})
        elif type(template_params) != dict:
            raise ValueError("Prompt params must be a dictionary")
        elif not all([type(k) == str for k in template_params.keys()]):
            raise ValueError("Prompt params keys must be strings")

        # ensure that keys are all strings
        for k in template_params.keys():
            if type(k) != str:
                raise ValueError("Prompt params keys must be strings")

        # try to render the template
        try:
            render = jinja_template.render(template_params)
        except Exception as e:
            raise ValueError(f"Error rendering system prompt: {e}")

        return render

    def resolve_prompt_template(self, template_name_or_template: str):
        try:
            template_path = os.path.join(os.path.dirname(__file__), "templates.yaml")
            with open(template_path, "r") as f:
                prompts = yaml.load(f, Loader=yaml.FullLoader)
                if template_name_or_template not in prompts:
                    return template_name_or_template
                return prompts[template_name_or_template]
        except:
            # return the template if exception
            return template_name_or_template

    def resolve_tools(self) -> Optional[list[dict]]:
        if self.tools is None:
            return None
        
        resolved_tools = []
        for tool in self.tools:
            if isinstance(tool, str):
                assert tool in all_agent_tools, \
                    f"Tool {tool} not found in any of the configs. Please include it in the config file under 'tools'."
                
                resolved_tools.append(all_agent_tools[tool])
            elif isinstance(tool, dict):
                resolved_tools.append(tool)
            else:
                raise ValueError(f"Invalid tool type. Expected str or dict, got {type(tool)}")
        
        return resolved_tools

    def validate_keys(self):
        # validate that the API keys are set
        model_key_validation = validate_environment(self.model)
        if not model_key_validation["keys_in_environment"]:
            raise ValueError(
                "Could not find the following API keys in the environment: {}".format(
                    ",".join(model_key_validation["missing_keys"])
                )
            )

    def copy(self) -> "AgentSettings":
        return AgentSettings(
            model=self.model,
            api_key=self.api_key,
            hyperparams=self.hyperparams,
            template_params=self.template_params,
            template=self.template,
            tools=self.tools,
            system_prompt=self.system_prompt,
            json_mode=self.json_mode,
            role=self.role,
        )

    def with_template_params(self, template_params: dict[str, str]) -> "AgentSettings":
        self.template_params = template_params
        return self

    def update(self, agent_settings: Any) -> None:
        if isinstance(agent_settings, dict):
            update_dict = agent_settings
        elif isinstance(agent_settings, AgentSettings):
            update_dict = agent_settings.__dict__
        else:
            raise TypeError(
                "agent_settings must be either a dictionary or an AgentSettings instance. Got {}".format(
                    type(agent_settings)
                )
            )

        valid_fields = {f.name for f in fields(self)}

        for key, value in update_dict.items():
            if key not in valid_fields:
                raise ValueError(f"Invalid field name: {key}")
            if value is not None:  # Only update if the value is not None
                setattr(self, key, value)


# holds the config for agents
all_agent_settings: dict[str, AgentSettings] = dict()
all_agent_tools: dict[str, list[dict]] = dict()

def print_system_prompt(prompt, role='assistant'):
    print(system_prompt_str(AgentSettings(system_prompt=prompt, role=role, model='')))

def system_prompt_str(agent_settings: AgentSettings):
    """Returns the system prompt for the given agent settings"""

    string = ""
    if agent_settings.role == "user":
        string = " ".join(
            (
                bcolors.HEADER + "\nUSER SYSTEM PROMPT\n\n",
                agent_settings.system_prompt,
                bcolors.ENDC,
            )
        )
    elif agent_settings.role == "assistant":
        string = " ".join(
            (
                bcolors.HEADER + "\nASSISTANT SYSTEM PROMPT\n\n",
                agent_settings.system_prompt,
                bcolors.ENDC,
            )
        )
    return string


def str_msgs(messages: list[OpenAIMessage] | OpenAIMessage):
    string = ""
    
    if isinstance(messages, OpenAIMessage):
        msgs_list = [messages]
    elif not messages or len(messages) == 0:
        return 'no messages found'
    else:
        msgs_list = messages
    
    for m in msgs_list:
        if m.role == "user":
            string += "\n" + " ".join(
                (bcolors.OKBLUE + "\n", m.role.upper(), "\n\n", m.content, bcolors.ENDC)
            )
        elif m.role == "assistant":
            string += "\n" + " ".join(
                (
                    bcolors.OKGREEN + "\n",
                    m.role.upper(),
                    "\n\n",
                    m.content,
                    bcolors.ENDC,
                )
            )
        elif m.role == "system":
            pass
    return string


def print_run_id(run_id):
    print("-" * 100)
    print("RUN ID:", run_id)
    print("-" * 100)


def swap_roles(messages: list[OpenAIMessage]) -> list[OpenAIMessage]:
    for message in messages:
        if message.role == "user":
            message.role = "assistant"
        elif message.role == "assistant":
            message.role = "user"
    return messages


def llm_call_resolve_agent_settings(
    agent_settings_or_name: Optional[Union[AgentSettings, dict, str]] = None,
    agent_settings: Optional[Union[AgentSettings, dict]] = None,
    agent_name: Optional[str] = None,
    **agent_settings_kwargs,
) -> AgentSettings:

    # assert all types
    assert isinstance(
        agent_settings_or_name, (AgentSettings, dict, str, type(None))
    ), f"agent_settings_or_name type {type(agent_settings_or_name)} is invalid"
    assert isinstance(
        agent_settings, (AgentSettings, dict, type(None))
    ), f"agent_settings type {type(agent_settings)} is invalid"
    assert isinstance(
        agent_name, (str, type(None))
    ), f"agent_name type {type(agent_name)} is invalid"

    agent_settings_to_use = None

    # First, check agent_settings_or_name
    if agent_settings_or_name is not None:
        if isinstance(agent_settings_or_name, AgentSettings):
            agent_settings_to_use = agent_settings_or_name
        elif isinstance(agent_settings_or_name, dict):
            agent_settings_to_use = AgentSettings(**agent_settings_or_name)
        elif isinstance(agent_settings_or_name, str):
            agent_settings_to_use = all_agent_settings[agent_settings_or_name]

    # Then, check agent_settings
    if agent_settings_to_use is None and agent_settings is not None:
        if isinstance(agent_settings, AgentSettings):
            agent_settings_to_use = agent_settings
        elif isinstance(agent_settings, dict):
            agent_settings_to_use = AgentSettings(**agent_settings)

    # Then, check agent_name
    if agent_settings_to_use is None and agent_name is not None:
        agent_settings_to_use = all_agent_settings[agent_name]

    # If still None, use default settings
    if agent_settings_to_use is None:
        agent_settings_to_use = AgentSettings(
            model="openai/gpt-4o-mini",
            role="assistant",
        )

    # Apply any additional kwargs
    for key, value in agent_settings_kwargs.items():
        # if the value and attribute are both dicts, merge them
        if hasattr(agent_settings_to_use, key) and \
            isinstance(value, dict) and \
            isinstance(getattr(agent_settings_to_use, key), dict):
            getattr(agent_settings_to_use, key).update(value)
        else:
            setattr(agent_settings_to_use, key, value)
    
    assert isinstance(
        agent_settings_to_use, AgentSettings
    ), f"agent_settings_to_use type {type(agent_settings_to_use)} is invalid"
    return agent_settings_to_use


def llm_call_get_completion_params(
    agent_settings: Optional[AgentSettings | dict[str, Any]] = None,
    messages: Optional[list[OpenAIMessage]] = None,
) -> dict:

    # ensure that messages is a list of OpenAIMessages
    for i, m in enumerate(messages or []):
        if not isinstance(m, OpenAIMessage):
            messages[i] = OpenAIMessage(**m)

    # resolve the prompt
    system_prompt = agent_settings.resolve_system_prompt()

    # validate the keys
    agent_settings.validate_keys()

    # insert / prepend / replace the system prompt
    if system_prompt is not None:
        if len(messages) == 0:
            messages = [OpenAIMessage(role="system", content=system_prompt)]
        elif messages[0].role != "system":
            messages.insert(0, OpenAIMessage(role="system", content=system_prompt))
        else:
            messages[0].content = system_prompt
            
    # assert len(messages) > 0, 'could not initialize messages'
        
    # add the init_messages only once at the beginning
    if agent_settings.init_messages:
        for i, m in enumerate(agent_settings.init_messages):
            if type(m) == dict:
                msg = OpenAIMessage(**m)
            elif type(m) == OpenAIMessage:
                msg = m
            else:
                raise ValueError(f"Invalid init_message type: {type(m)}")
            
            if msg.role != "system":
                # assume that if the exact same messages are present already at the same positions,
                # we don't need to add them again
                if messages[i + 1] != msg:
                    messages.insert(i + 1, msg)

    # swap roles for user
    if agent_settings.role == "user":
        messages = swap_roles(messages)

    # get the response format
    response_format = agent_settings.resolve_response_format()

    # resolve hyperparams
    hyperparams = agent_settings.hyperparams or dict()

    # resolve tools
    tools = agent_settings.resolve_tools()

    # resolve api_key
    api_key = None
    if agent_settings.api_key:
        os.getenv(agent_settings.api_key)

    # convert messages to dict
    messages_to_llm = [m.__dict__() for m in messages]

    return {
        "model": agent_settings.model,
        "api_key": api_key,
        "messages": messages_to_llm,
        "tools": tools,
        "response_format": response_format,
        **hyperparams,
    }


def llm_call_post_process_response(
    agent_settings: AgentSettings,
    messages: list[OpenAIMessage],
    response: Optional[ModelResponse | Exception],
) -> OpenAIMessage:

    # unswap roles for user
    if agent_settings.role == "user":
        messages = swap_roles(messages)

    # process empty or error responses
    assert response is not None, "Received empty response."
    if isinstance(response, Exception):
        print(
            f"API call for {agent_settings.model} failed: {response}. Returning string 'error' and continuing."
        )
        # some models require an alternate user / assistant dialog
        if len(messages) == 0 or messages[-1].role != "user":
            return OpenAIMessage(role="user", content="error")
        else:
            return OpenAIMessage(role="assistant", content="error")

    # process the response
    response_choice = response.choices[0]
    if response_choice.finish_reason == 'stop':
        raw_message = response_choice.message
        response_messages_role = agent_settings.role or raw_message["role"]
        response_message = OpenAIMessage(
            role=response_messages_role, content=raw_message.content
        )
        
        if agent_settings.json_mode:
            response_message.content = json.loads(response_message.content)

    else:
        raise ValueError(f'Invalid finish reason: {response_choice.finish_reason}')

    return response_message

async def allm_tool_call(
    response: ModelResponse,
    tool_funcs: list[Callable],
) -> Optional[list[OpenAIMessage]]:
    
    response_choice: list[Choices] = response.choices[0]
    
    if response_choice.finish_reason != 'tool_calls' or response_choice.message.tool_calls is None:
        return None
    
    tool_func_map = dict()
    if tool_funcs is not None:
        for tool_func in tool_funcs:
            assert isinstance(tool_func, Callable), f"Tool function {tool_func} is not callable"
            assert tool_func.__name__ is not None, f"Tool function {tool_func} has no name"
            tool_func_map[tool_func.__name__] = tool_func
            
    def is_coroutine(func: Callable) -> bool:
        return asyncio.iscoroutinefunction(func) or \
            isinstance(func, aevaluator)
        
    async def execute_tool_call(tool_call):
        assert "id" in tool_call, "Tool call id not found"
        assert "function" in tool_call, "Tool call function not found"
        assert tool_call.function.name is not None, "Tool call function name not found"
        assert tool_call.function.arguments is not None, "Tool call arguments not found"
        function_name = tool_call.function.name
        
        # parse the arguments
        try:
            function_args = json.loads(tool_call.function.arguments)
        except Exception as e:
            print(f'Error parsing tool call arguments for {function_name}: {e}')
            response_content = f'Error parsing tool call arguments for {function_name}: {e}'
            return OpenAIMessage(
                role='tool',
                name=function_name,
                tool_call_id=tool_call.id,
                content=response_content
            )
        
        # call the tool
        try:
            response_content = None
            start_time = asyncio.get_event_loop().time()
            
            if function_name in evaluator.all_evaluators:
                if is_coroutine(evaluator[function_name]):
                    function_response = await evaluator[function_name](**function_args)
                else:
                    function_response = await asyncio.to_thread(evaluator[function_name], **function_args)
            elif function_name in tool_func_map:
                if is_coroutine(tool_func_map[function_name]):
                    function_response = await tool_func_map[function_name](**function_args)
                else:
                    function_response = await asyncio.to_thread(tool_func_map[function_name], **function_args)
            else:
                raise ValueError(f"Function {function_name} implementation not found in tool_funcs or evaluator")
            
            end_time = asyncio.get_event_loop().time()
            
            # append the tool call to the messages
            print('Successfully called tool', function_name, f'in {end_time - start_time:.3f} seconds.')
            response_content = str(function_response)
        
        except Exception as e:
            print(f'Error calling tool {function_name}: {e}')
            response_content = f'Error calling tool {function_name}: {e}. Do not retry.'
            
        finally:
            if response_content is None:
                response_content = f'Could not call tool {function_name}. Do not retry.'
            
            return OpenAIMessage(
                role='tool',
                name=function_name,
                tool_call_id=tool_call.id,
                content=response_content
            )
    
    # execute all the tool calls in parallel
    response_messages = await asyncio.gather(
        *[
            execute_tool_call(tool_call) 
            for tool_call in response_choice.message.tool_calls
        ]
    )

    return response_messages

def llm_tool_call(
    response: ModelResponse,
    tool_funcs: list[Callable],
) -> Optional[list[OpenAIMessage]]:
    
    response_choice = response.choices[0]
    
    if response_choice.finish_reason != 'tool_calls' or response_choice.message.tool_calls is None:
        return None
    
    tool_func_map = dict()
    if tool_funcs is not None:
        for tool_func in tool_funcs:
            assert isinstance(tool_func, Callable), f"Tool function {tool_func} is not callable"
            assert tool_func.__name__ is not None, f"Tool function {tool_func} has no name"
            tool_func_map[tool_func.__name__] = tool_func
        
    response_messages = []
    
    for tool_call in response_choice.message.tool_calls:
        assert "id" in tool_call, "Tool call id not found"
        assert "function" in tool_call, "Tool call function not found"
        assert tool_call.function.name is not None, "Tool call function name not found"
        assert tool_call.function.arguments is not None, "Tool call arguments not found"
        function_name = tool_call.function.name
        
        try:
            function_args = json.loads(tool_call.function.arguments)
            
            # call the tool
            start_time = time.time()
            
            if function_name in evaluator.all_evaluators:
                assert not asyncio.iscoroutine(evaluator[function_name]), f"Tool function {function_name} is a coroutine"
                function_response = evaluator[function_name](**function_args)
            elif function_name in tool_func_map:
                assert not asyncio.iscoroutine(tool_func_map[function_name]), f"Tool function {function_name} is a coroutine"
                function_response = tool_func_map[function_name](**function_args)
            else:
                raise ValueError(f"Function {function_name} implementation not found in tool_funcs or evaluator")
            
            end_time = time.time()
            
            # append the tool call to the messages
            response_messages.append(OpenAIMessage(
                role='tool',
                name=function_name,
                tool_call_id=tool_call.id,
                content=function_response
            ))
            print('Successfully called tool', function_name, f'in {end_time - start_time:.3f} seconds.')
            
        except Exception as e:
            print(f'Error calling tool {function_name}: {e}')
            
            response_messages.append(OpenAIMessage(
                role='tool',
                name=function_name,
                tool_call_id=tool_call.id,
                content=f'Error calling tool {function_name}: {e}. Do not retry.'
            ))
    
    return response_messages

def llm_messages_call(
    agent_settings_or_name: Optional[Union[AgentSettings, dict, str]] = None,
    agent_settings: Optional[Union[AgentSettings, dict]] = None,
    agent_name: Optional[str] = None,
    messages: list[OpenAIMessage] | list[dict[str, str]] = [],
    tool_funcs: Optional[list[Callable]] = None,
    **agent_settings_kwargs,
) -> OpenAIMessage:
    
    # resolve the agent settings
    agent_settings = llm_call_resolve_agent_settings(
        agent_settings_or_name=agent_settings_or_name,
        agent_settings=agent_settings,
        agent_name=agent_name,
        **agent_settings_kwargs,
    )

    # get the params
    params = llm_call_get_completion_params(agent_settings, messages)

    # call the LLM using the router
    response: ModelResponse = router.completion(**params)
    
    # keep calling the LLM until the tool calls are resolved
    while response.choices[0].finish_reason == 'tool_calls' and \
        response.choices[0].message.tool_calls:
        
        # append the tool call message
        if response.choices[0].finish_reason == 'tool_calls':
            params['messages'].append(response.choices[0].message)
        
        # call the tool
        messages: Optional[list[OpenAIMessage]] = llm_tool_call(response, tool_funcs)
        if messages is None or len(messages) == 0:
            break
        
        # append the tool call responses
        params['messages'].extend([m.__dict__() for m in messages])
        
        # call the LLM again
        response: ModelResponse = router.completion(**params)
    
    # post process the response
    message: OpenAIMessage = llm_call_post_process_response(
        agent_settings, messages, response
    )

    return message


async def allm_messages_call(
    agent_settings_or_name: Optional[Union[AgentSettings, dict, str]] = None,
    messages: list[OpenAIMessage] | list[dict[str, str]] = [],
    agent_settings: Optional[Union[AgentSettings, dict]] = None,
    agent_name: Optional[str] = None,
    tool_funcs: Optional[list[Callable]] = None,
    **agent_settings_kwargs,
) -> OpenAIMessage:
    """Make an LLM call with provided agent_settings and messages"""

    # resolve the agent settings
    agent_settings = llm_call_resolve_agent_settings(
        agent_settings_or_name=agent_settings_or_name,
        agent_settings=agent_settings,
        agent_name=agent_name,
        **agent_settings_kwargs,
    )

    # get the params
    params = llm_call_get_completion_params(agent_settings, messages)

    # call the LLM using the router
    response: ModelResponse = await router.acompletion(**params)
    
    # keep calling the LLM until the tool calls are resolved
    while response.choices[0].finish_reason == 'tool_calls' and \
        response.choices[0].message.tool_calls:
        
        # append the tool call message
        if response.choices[0].finish_reason == 'tool_calls':
            params['messages'].append(response.choices[0].message)
        
        # call the tool
        messages: Optional[list[OpenAIMessage]] = await allm_tool_call(response, tool_funcs)
        if messages is None or len(messages) == 0:
            break
        
        # append the tool call responses to the messages
        params['messages'].extend([m.__dict__() for m in messages])
        
        # call the LLM again
        response: ModelResponse = await router.acompletion(**params)
        
    # post process the response
    message: OpenAIMessage = llm_call_post_process_response(
        agent_settings, messages, response
    )

    return message


async def aembed_text(text: str, **kwargs):
    if "dimensions" not in kwargs:
        kwargs["dimensions"] = 512
    response = await aembedding("text-embedding-3-small", input=text, **kwargs)
    return response


def messages_to_string(messages: list[OpenAIMessage]) -> str:
    """Convert a list of messages to a string"""
    return "\n".join([m.role + ":\n" + m.content for m in messages])


def get_realign_llm_utils_globals():
    try:
        from realign.llm_utils import allm_messages_call
    except ImportError:
        raise

    return {
        "allm_messages_call": allm_messages_call,
    }
