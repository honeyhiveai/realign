from realign.types import ModelSettings, OpenAIMessage, RunData, EvalResult
from realign.router import Router
from typing import Any
from litellm import aembedding

import os
import json
import asyncio
from functools import wraps


# initialize the request router
router = Router()

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

def print_chat(messages: list[OpenAIMessage]):
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

def print_evals(evals: list[EvalResult]):
    print(bcolors.WARNING)
    for e in evals:
        print(e)
        print('- ' * 50)
    print(bcolors.ENDC)

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
    
    # validate the keys
    model_settings.validate_keys()    
    
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
    response = router.completion(**params)

    # post process the response
    message: OpenAIMessage = llm_call_post_process_response(model_settings, messages, response)
    
    return message

async def allm_messages_call(model_settings: ModelSettings, messages: list[OpenAIMessage] = []) -> OpenAIMessage:
    '''Make an LLM call with the messages provided'''

    # get the params
    params = llm_call_get_completion_params(model_settings, messages)

    # call the LLM
    response = await router.acompletion(**params)
    
    # post process the response
    message: OpenAIMessage = llm_call_post_process_response(model_settings, messages, response)
    
    return message

async def aembed_text(text: str, **kwargs) -> str:
    if 'dimensions' not in kwargs:
        kwargs['dimensions'] = 512
    response = await aembedding('text-embedding-3-small', input=text, **kwargs)
    return response

def messages_to_string(messages: list[OpenAIMessage]) -> str:
    '''Convert a list of messages to a string'''
    return '\n'.join([m.role + ':\n' + m.content for m in messages])


# evaluator decorator to wrap an app scorer inside a EvalResult object
# TODO: composable evaluators
def evaluator(eval_func=None, *, repeat=1, embed_explanation=True) -> EvalResult:
    def decorator(func):
        @wraps(eval_func)
        async def wrapper(run_data: RunData, *args, **kwargs):
            async def single_run():
                if asyncio.iscoroutinefunction(func):
                    # If the eval_func is already a coroutine function, just await it
                    response = await func(run_data.final_state, 
                                          *args,
                                          **kwargs
                                          )
                else:
                    # If it's a regular function, run it in a thread pool
                    response = await asyncio.to_thread(func, 
                                                       run_data.final_state,
                                                       *args,
                                                       **kwargs
                                                        )

                # verify that the eval_func response is a tuple of 3 elements (score: Any, result: bool | None, explanation: str | None)
                if not 2 <= len(response) <= 3 or \
                    type(response) not in [list, tuple] or \
                        type(response[1]) not in [bool, type(None)] or \
                            (len(response) == 3 and type(response[2]) not in [str, type(None)]):
                    raise ValueError('Evaluator response must be a tuple of 2 elements, the score of type Any and result of type bool | None')

                # unpack results
                score = None
                result = None
                explanation = None
                if len(response) == 3 and response[2] is not None:
                    score, result, explanation = response
                    if embed_explanation:
                        embedding = await aembed_text(explanation)
                    return (score, result, explanation, embedding)

                score, result = response
                return (score, result, None, None)

            assert repeat >= 0, 'Repeat must be greater than 0'

            if repeat == 0:
                return None

            if repeat > 1:
                tasks = [single_run() for _ in range(repeat)]
                score_result_tuples = await asyncio.gather(*tasks)
                scores, results, explanations, embeddings = zip(*score_result_tuples)

                # for repeats, return the full array of scores and results
                return EvalResult(
                    scores,
                    results,
                    explanations,
                    embeddings,
                    run_data,
                    func.__name__,
                    repeat
                )
            else:
                # for single runs, return a single score and result
                score, result, explanation, embedding = await single_run()
                return EvalResult(
                    score,
                    result,
                    explanation,
                    embedding,
                    run_data,
                    func.__name__,
                    repeat
                )

        return wrapper

    if eval_func is None:
        return decorator
    else:
        return decorator(eval_func)