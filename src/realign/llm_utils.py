import os
import json
import asyncio
from dataclasses import dataclass
from typing import Any, Optional, Callable, Union, Coroutine
import hashlib
import yaml


from jinja2 import Template

from litellm import ModelResponse, aembedding, acompletion, validate_environment
import litellm

from realign.router import Router
from realign.configs import load_yaml_settings
from realign.utils import bcolors, run_async


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

    def __getitem__(self, key: str):
        if key == 'role':
            return self.role
        elif key == 'content':
            return self.content
        else:
            raise KeyError(key)

    def __dict__(self):
        return {"role": str(self.role), "content": str(self.content)}

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

    # user or assistant
    role: str = "assistant"
    
    # initialize the messages
    # system messages will be ignored, since they must be defined in the template / system_prompt
    init_messages: Optional[list[OpenAIMessage]] = None

    def resolve_response_format(self) -> str:
        if self.json_mode:
            return {"type": "json_object"}
        return None

    def resolve_system_prompt(self) -> str:
        prompt_to_render = ""
        system_prompt = self.system_prompt
        template = self.template
        if system_prompt is None:
            if template is None:
                raise ValueError(
                    "Either system_prompt or template must be provided in the model settings"
                )
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
            system_prompt=self.system_prompt,
            json_mode=self.json_mode,
            role=self.role,
        )

    def with_template_params(self, template_params: dict[str, str]) -> "AgentSettings":
        self.template_params = template_params
        return self

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
            agent_settings_to_use = get_agent_settings(
                agent_name=agent_settings_or_name
            )

    # Then, check agent_settings
    if agent_settings_to_use is None and agent_settings is not None:
        if isinstance(agent_settings, AgentSettings):
            agent_settings_to_use = agent_settings
        elif isinstance(agent_settings, dict):
            agent_settings_to_use = AgentSettings(**agent_settings)

    # Then, check agent_name
    if agent_settings_to_use is None and agent_name is not None:
        agent_settings_to_use = get_agent_settings(agent_name=agent_name)

    # If still None, use default settings
    if agent_settings_to_use is None:
        agent_settings_to_use = AgentSettings(
            model="openai/gpt-4o-mini",
            role="assistant",
        )

    # Apply any additional kwargs
    for key, value in agent_settings_kwargs.items():
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
    if len(messages) == 0:
        messages = [OpenAIMessage(role="system", content=system_prompt)]
    elif messages[0].role != "system":
        messages.insert(0, OpenAIMessage(role="system", content=system_prompt))
    else:
        messages[0].content = system_prompt
        
    assert len(messages) > 0 and messages[0].role == "system", 'could not initialize messages'
        
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

    raw_message = response.choices[0].message
    response_messages_role = agent_settings.role or raw_message["role"]
    response_message = OpenAIMessage(
        role=response_messages_role, content=raw_message["content"]
    )

    if agent_settings.json_mode:
        response_message.content = json.loads(response_message.content)

    return response_message

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        raise

def llm_messages_call(
    agent_settings_or_name: Optional[Union[AgentSettings, dict, str]] = None,
    agent_settings: Optional[Union[AgentSettings, dict]] = None,
    agent_name: Optional[str] = None,
    messages: list[OpenAIMessage] | list[dict[str, str]] = [],
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

    # post process the response
    message: OpenAIMessage = llm_call_post_process_response(
        agent_settings, messages, response
    )

    return message


async def allm_messages_call(
    agent_settings_or_name: Optional[Union[AgentSettings, dict, str]] = None,
    agent_settings: Optional[Union[AgentSettings, dict]] = None,
    agent_name: Optional[str] = None,
    messages: list[OpenAIMessage] | list[dict[str, str]] = [],
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


def get_agent_settings(
    yaml_file: Optional[str] = None, agent_name: Optional[str] = None
) -> dict[str, AgentSettings] | AgentSettings:

    parsed_yaml = load_yaml_settings(yaml_file)

    if not isinstance(parsed_yaml, dict) or "llm_agents" not in parsed_yaml:
        raise ValueError(
            "Invalid YAML structure. Expected 'llm_agents' key at the root level."
        )

    assert isinstance(
        parsed_yaml["llm_agents"], dict
    ), "llm_agents must be a dictionary"

    agent_settings = {}
    for _agent_name, settings in parsed_yaml["llm_agents"].items():
        agent_settings[_agent_name] = AgentSettings(**settings)

    if agent_name is not None:
        if agent_name not in agent_settings:
            raise ValueError(f"Agent '{agent_name}' not found in 'llm_agents' section.")
        return agent_settings[agent_name]

    return agent_settings


def get_realign_llm_utils_globals():
    try:
        from realign.llm_utils import allm_messages_call
    except ImportError:
        raise

    return {
        "allm_messages_call": allm_messages_call,
    }
