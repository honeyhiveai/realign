from realign.types import ModelSettings, OpenAIMessage
from typing import Any
from litellm import completion, acompletion
import os
import json


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_system_prompt(model_settings: ModelSettings):
    if model_settings.role == 'user':
        print(bcolors.HEADER + '\nUSER SYSTEM PROMPT\n\n', model_settings.system_prompt, bcolors.ENDC)
    elif model_settings.role == 'assistant':
        print(bcolors.HEADER + '\nASSISTANT SYSTEM PROMPT\n\n', model_settings.system_prompt, bcolors.ENDC)

def print_chat(messages):
    for m in messages:
        if m.role == 'user':
            print(bcolors.OKBLUE + '\n', m.role.upper(), '\n\n', m.content, bcolors.ENDC)
        elif m.role == 'assistant':
            print(bcolors.OKGREEN + '\n', m.role.upper(), '\n\n', m.content, bcolors.ENDC)
        elif m.role == 'system':
            pass
        
def print_run_id(run_id):
    print('-' * 100)
    print('RUN ID:',run_id)
    print('-' * 100)
    
def swap_roles(messages: list[OpenAIMessage]) -> list[OpenAIMessage]:
    for message in messages:
        if message.role == 'user':
            message.role = 'assistant'
        elif message.role == 'assistant':
            message.role = 'user'
    return messages

def llm_call_get_completion_params(model_settings: ModelSettings, messages: list[OpenAIMessage]) -> dict:
        
    # resolve the prompt
    system_prompt = model_settings.resolve_system_prompt()
    
    # insert the system prompt
    if len(messages) == 0:
        messages = [OpenAIMessage(role='system', content=system_prompt)]
    elif messages[0].role != 'system':
        messages.insert(0, OpenAIMessage(role='system', content=system_prompt))
    else:
        messages[0].content = system_prompt
        
    # swap roles for user
    if model_settings.role == 'user':
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
        'model': model_settings.model,
        'api_key': api_key,
        'messages': messages_to_llm,
        'response_format': response_format,
        **hyperparams,
    }
    
def llm_call_post_process_response(model_settings: ModelSettings, messages: list[OpenAIMessage], response: Any) -> Any:
    
    # unswap roles for user
    if model_settings.role == 'user':
        messages = swap_roles(messages)

    # process the message
    raw_message = response.choices[0].message
    response_message = OpenAIMessage(role=raw_message['role'], content=raw_message['content'])
    if model_settings.json_mode:
        response_message.content = json.loads(response_message.content)

    return response_message

def llm_messages_call(model_settings: ModelSettings, messages: list[OpenAIMessage] = []) -> OpenAIMessage:
    '''Make an LLM call with the messages provided'''

    # get the params
    params = llm_call_get_completion_params(model_settings, messages)

    # call the LLM
    response = completion(**params)
    
    # post process the response
    message: OpenAIMessage = llm_call_post_process_response(model_settings, messages, response)
    
    return message

async def allm_messages_call(model_settings: ModelSettings, messages: list[OpenAIMessage] = []) -> OpenAIMessage:
    '''Make an LLM call with the messages provided'''

    # get the params
    params = llm_call_get_completion_params(model_settings, messages)

    # call the LLM
    response = await acompletion(**params)
    
    # post process the response
    message: OpenAIMessage = llm_call_post_process_response(model_settings, messages, response)
    
    return message

def messages_to_string(messages: list[OpenAIMessage]) -> str:
    '''Convert a list of messages to a string'''
    return '\n'.join([m.role + ':\n' + m.content for m in messages])