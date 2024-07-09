from agentsim.evaluators.targets import NumericRangeTarget, BaseTarget, ContainsTarget
from agentsim.Simulator import Simulator
from agentsim.types import EvaluatorConfig
from typing import Any
import os
import inspect
from jinja2 import Template

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
    
def print_system_prompt(sim):
    user_sys_prompt = sim.sim_agent.config.system_prompt
    print(bcolors.HEADER + '\nUSER SYSTEM PROMPT\n\n', user_sys_prompt, bcolors.ENDC)
    
    app_sys_prompt = sim.app_agent.config.system_prompt
    print(bcolors.HEADER + '\nAPP SYSTEM PROMPT\n\n', app_sys_prompt, bcolors.ENDC)

def print_chat(messages):
    for m in messages:
        if m["role"] == 'user':
            print(bcolors.OKBLUE + '\n', m["role"].upper(), '\n\n', m["content"], bcolors.ENDC)
        elif m["role"] == 'assistant':
            print(bcolors.OKGREEN + '\n', m["role"].upper(), '\n\n', m["content"], bcolors.ENDC)
        elif m["role"] == 'system':
            pass

def get_evaluator_config_from_env(eval_func) -> EvaluatorConfig:
    
    # load the config from the environment variable
    str_yaml = os.getenv('AGENTSIM_CONFIG')
    simulator = Simulator(str_yaml=str_yaml, config_only=True)
    
    # get the evaluator config
    eval_config = simulator.agentsim_config.evaluators[eval_func]
    return eval_config

def guardrail_evaluator(in_range: str, target) -> BaseTarget:
    if in_range == 'numeric_range':
        target = NumericRangeTarget(target)
    elif in_range == 'contains':
        target = ContainsTarget(target)
    else:
        target = BaseTarget(target)
    return target

def check_guardrail(score: Any, eval_func=None) -> bool:
    '''Check if the score is in the target range'''
    
    # if no eval_func, load the parent function that called this function
    if not eval_func:
        eval_func = inspect.stack()[1].function
    
    # get the evaluator config
    eval_config = get_evaluator_config_from_env(eval_func)

    # set the callable target object
    evaluate_target = guardrail_evaluator(eval_config.in_range, eval_config.target)

    # run the evaluation
    result = evaluate_target(score)

    return result


def llm_eval_call(template_params: dict[str, Any]) -> Any:
    '''Evaluate using an LLM'''
    
    from litellm import completion
    import json
    
    # load the parent function that called this function
    eval_func = inspect.stack()[1].function
    
    # get the evaluator config
    eval_config = get_evaluator_config_from_env(eval_func)
    
    # get the model settings
    model_settings = eval_config.model_settings
    
    # make the completion call
    eval_prompt_template = Template(model_settings.system_prompt)
    eval_prompt = eval_prompt_template.render(template_params)
    
    # get the response
    response_format = { 'type': "json_object" } if model_settings.json_mode else None
    response = completion(
        model=model_settings.model,
        api_key=os.getenv(model_settings.api_key),
        messages=[
            {
                'role': 'user',
                'content': eval_prompt
            }
        ],
        response_format=response_format,
        **model_settings.hyperparams
    )

    if model_settings.json_mode:
        return json.loads(response.choices[0].message['content'])

    return response.choices[0].message['content']


def text_format_messages(messages):
    '''Format messages for LLM evaluation'''
    
    # format the messages into a giant string. Don't include system messages
    str_messages = ''
    for message in messages:
        if message['role'] == 'user':
            str_messages += message['content'] + ' '
        elif message['role'] == 'assistant':
            str_messages += message['content'] + ' '
    
    return str_messages
    
def evaluations_dataframe(evaluations):
    '''Create a dataframe from evaluations'''
    
    import pandas as pd
    
    return pd.DataFrame([eval.to_dict() for eval in evaluations])