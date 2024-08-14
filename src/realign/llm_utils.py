import os
import json
import asyncio
from dataclasses import dataclass

from typing import Any, Optional
from litellm import ModelResponse, aembedding, acompletion
import litellm

from realign.types import OpenAIMessage, RunData, EvalResult, bcolors
from realign.router import Router
from realign.config import get_model_settings, ModelSettings

# this flag helps litellm modify params to ensure that model-specific requirements are met
litellm.modify_params = True

# initialize the request router
router = Router()

@dataclass
class State:
    messages: list[OpenAIMessage]
    
    def __init__(self):
        self.messages = []
        
    def __repr__(self) -> str:
        return str_msgs(self.messages[1:])


def system_prompt_str(model_settings: ModelSettings):
    string = ''
    if model_settings.role == "user":
        string = ' '.join(
            (bcolors.HEADER + "\nUSER SYSTEM PROMPT\n\n", 
            model_settings.system_prompt, 
            bcolors.ENDC)
        )
    elif model_settings.role == "assistant":
        string = ' '.join(
            (bcolors.HEADER + "\nASSISTANT SYSTEM PROMPT\n\n",
            model_settings.system_prompt,
            bcolors.ENDC)
        )
    return string

def str_msgs(messages: list[OpenAIMessage]):
    string = ''
    for m in messages:
        if m.role == "user":
            string += '\n' + ' '.join(
                (bcolors.OKBLUE + "\n", m.role.upper(), "\n\n", m.content, bcolors.ENDC)
            )
        elif m.role == "assistant":
            string += '\n' + ' '.join(
                (bcolors.OKGREEN + "\n", m.role.upper(), "\n\n", m.content, bcolors.ENDC)
            )
        elif m.role == "system":
            pass
    return string

def print_run_id(run_id):
    print("-" * 100)
    print("RUN ID:", run_id)
    print("-" * 100)


def print_evals(evals: list[EvalResult]):
    print(bcolors.WARNING)
    for e in evals:
        print(e)
        print("- " * 50)
    print(bcolors.ENDC)


def swap_roles(messages: list[OpenAIMessage]) -> list[OpenAIMessage]:
    for message in messages:
        if message.role == "user":
            message.role = "assistant"
        elif message.role == "assistant":
            message.role = "user"
    return messages

def llm_call_get_completion_params(
    model_settings: Optional[ModelSettings | dict[str, Any]] = None, 
    messages: Optional[list[OpenAIMessage]] = None
) -> dict:
    
    # ensure that messages is a list of OpenAIMessages
    for i, m in enumerate(messages or []):
        if not isinstance(m, OpenAIMessage):
            messages[i] = OpenAIMessage(**m)

    # resolve the prompt
    system_prompt = model_settings.resolve_system_prompt()

    # validate the keys
    model_settings.validate_keys()
    
    if len(messages) == 0:
        messages = [OpenAIMessage(role="system", content=system_prompt)]
    elif messages[0].role != "system":
        messages.insert(0, OpenAIMessage(role="system", content=system_prompt))
    else:
        messages[0].content = system_prompt

    # swap roles for user
    if model_settings.role == "user":
        messages = swap_roles(messages)

    # get the response format
    response_format = model_settings.resolve_response_format()

    # resolve hyperparams
    hyperparams = model_settings.hyperparams or dict()

    # resolve api_key
    api_key = None
    if model_settings.api_key:
        os.getenv(model_settings.api_key)

    # convert messages to dict
    messages_to_llm = [m.__dict__() for m in messages]

    return {
        "model": model_settings.model,
        "api_key": api_key,
        "messages": messages_to_llm,
        "response_format": response_format,
        **hyperparams,
    }

def llm_call_post_process_response(
    model_settings: ModelSettings, messages: list[OpenAIMessage], response: Optional[ModelResponse | Exception]
) -> OpenAIMessage:

    # unswap roles for user
    if model_settings.role == "user":
        messages = swap_roles(messages)

    # process empty or error responses
    assert response is not None, "Received empty response."
    if isinstance(response, Exception):
        print(f"API call for {model_settings.model} failed: {response}. Returning string 'error' and continuing.")
        # some models require an alternate user / assistant dialog
        if len(messages) == 0 or messages[-1].role != "user":
            return OpenAIMessage(role="user", content="error")
        else:
            return OpenAIMessage(role='assistant', content="error")
    
    raw_message = response.choices[0].message
    response_message = OpenAIMessage(
        role=raw_message["role"], content=raw_message["content"]
    )
    if model_settings.json_mode:
        response_message.content = json.loads(response_message.content)

    return response_message


def llm_messages_call(
    model_settings: Optional[ModelSettings], messages: list[OpenAIMessage] = []
) -> OpenAIMessage:
    raise NotImplementedError

async def allm_messages_call(
    model_settings: Optional[ModelSettings | str] = None, 
    messages: list[OpenAIMessage] | list[dict[str, str]] = [],
    **model_settings_kwargs,
) -> OpenAIMessage:
    """Make an LLM call with provided model_settings and messages"""
    
    # resolve the model_settings
    if model_settings is None:
        model_settings = ModelSettings(
            model="openai/gpt-4o-mini",
            system_prompt="Be a good assistant.",
            role="assistant",
        )
    elif isinstance(model_settings, dict):
        model_settings = ModelSettings(**model_settings)
    elif isinstance(model_settings, str):
        model_settings: ModelSettings = get_model_settings(agent_name=model_settings)
        
    if model_settings_kwargs:
        for setting, value in model_settings_kwargs.items():
            # for template_params and hyperparams, we need to merge
            if isinstance(getattr(model_settings, setting), dict) and isinstance(value, dict):
                current_setting: dict = getattr(model_settings, setting)
                current_setting.update(value)
            # for everything else, just set the value
            else:
                setattr(model_settings, setting, value)

    assert type(model_settings) == ModelSettings, "model_settings type is invalid"

    # get the params
    params = llm_call_get_completion_params(model_settings, messages)

    # call the LLM using the router
    response: ModelResponse = await router.acompletion(**params)

    # post process the response
    message: OpenAIMessage = llm_call_post_process_response(
        model_settings, messages, response
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
