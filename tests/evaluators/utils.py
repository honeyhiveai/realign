from agentsim.targets import NumericRangeTarget, BaseTarget, ContainsTarget
from agentsim.Simulator import Simulator
# from agentsim.types.types import EvaluatorConfig, ModelSettings
from agentsim import prompts
from typing import Any
import os
import inspect
from jinja2 import Template
from litellm import completion
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
    
def print_system_prompt(sim, app_agents=None):
    user_sys_prompt = sim.sim_agent.config.system_prompt
    print(bcolors.HEADER + '\nUSER SYSTEM PROMPT\n\n', user_sys_prompt, bcolors.ENDC)
    
    if app_agents:
        for app_agent in app_agents:
            app_sys_prompt = sim.app_agents[app_agent].config.system_prompt
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
    
    if not(eval_config.in_range and eval_config.target):
        return None

    # set the callable target object
    evaluate_target = guardrail_evaluator(eval_config.in_range, eval_config.target)

    # run the evaluation
    result = evaluate_target(score)
        
    return result

def resolve_template(template_name: str):
    if template_name == 'rating_5_star':
        return prompts.rating_5_star
    raise ValueError("Template not found")

def resolve_system_prompt(model_settings: ModelSettings) -> str:
    prompt_to_render = ''
    system_prompt = model_settings.system_prompt
    template = model_settings.template
    if template == '' and system_prompt == '':
        raise ValueError("Either system_prompt or template must be provided in the model settings")

    # use system prompt if it is provided
    if system_prompt != '':
        prompt_to_render = system_prompt
    else:
        prompt_to_render = resolve_template(template)
    
    jinja_template = Template(prompt_to_render)
    prompt_params = model_settings.prompt_params
    if prompt_params is None:
        return jinja_template.render({})
    elif type(prompt_params) != dict:
        raise ValueError("Prompt params must be a dictionary")
    elif not all([type(k) == str for k in prompt_params.keys()]):
        raise ValueError("Prompt params keys must be strings")
    
    # ensure that values are all strings
    for k, v in prompt_params.items():
        if type(k) != str:
            raise ValueError("Prompt params keys must be strings")
        if type(v) != str:
            prompt_params[k] = str(v)
    
    # try to render the template
    try:
        render = jinja_template.render(prompt_params)
    except Exception as e:
        raise ValueError(f"Error rendering system prompt: {e}")
    
    return render
        

def llm_eval_call(eval_func=None) -> Any:
    '''Evaluate using an LLM'''

    # load the parent function that called this function
    if eval_func is None:
        eval_func = inspect.stack()[1].function
    
    # get the evaluator config
    eval_config = get_evaluator_config_from_env(eval_func)
    
    # get the model settings
    model_settings = eval_config.model_settings
    
    # resolve the prompt
    eval_prompt = resolve_system_prompt(model_settings)
    
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


def llm_rating_evaluator(messages, eval_func=None):
    
    # if no eval_func, load the parent function that called this function
    if eval_func is None:
        eval_func = inspect.stack()[1].function
    
    # get the toxicity score
    score = llm_eval_call({
        'messages': str(messages)
    }, eval_func)['rating']
    
    # check if the score is in the target range
    result = check_guardrail(score, eval_func)
    
    return score, result

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