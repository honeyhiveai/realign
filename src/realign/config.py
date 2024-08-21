import yaml
import os
from typing import Optional

import realign


DEFAULT_CONFIG_PATH = 'src/realign/defaults.yaml'

# Export the config_path property for easy access
config_path = DEFAULT_CONFIG_PATH

def load_config(path):
    global config_path
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    
    # if path unchanged, return
    if config_path == path:
        return
    
    # check if valid yaml
    config_content = None
    with open(path) as f:
        try:
            config_content = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
        
        if not ('llm_agents' in config_content and 'evaluators' in config_content):
            raise ValueError(f"Invalid YAML structure. Expected 'llm_agents' or 'evaluators' keys at the root level.")
    

    # TODO: cleaner solution, circular import
    from realign.evaluators import evaluator, get_eval_settings
    
    all_eval_settings, all_eval_kwargs = get_eval_settings(yaml_file=path)
    
    evaluator.all_eval_settings.update(all_eval_settings)
    evaluator.all_eval_kwargs.update(all_eval_kwargs)
    
    # update their settings based on the config
    for eval_name in evaluator.all_evaluators.keys():   
        if eval_name in all_eval_settings:
            
            evaluator.all_evaluators[eval_name].eval_settings.update(all_eval_settings[eval_name])
        
        if eval_name in all_eval_kwargs:
            evaluator.all_evaluators[eval_name].eval_kwargs.update(all_eval_kwargs[eval_name])
    
    config_path = path
    
    print('Loaded config file:', path)


def load_yaml_settings(yaml_file: Optional[str] = None) -> dict[str, dict]:
    
    def resolve_config_path() -> Optional[str]:
        config_path = realign.config_path
        if type(config_path) == str:
            return config_path
        elif type(config_path) == property:
            return config_path.fget()
        return None
    
    # look for config.yaml in the current directory
    yaml_file = yaml_file or \
                os.getenv('REALIGN_CONFIG_PATH') or \
                resolve_config_path()
    
    if yaml_file is None:
        raise ValueError("No config file specified. Please set the REALIGN_CONFIG_PATH environment variable or pass in a config file path.")

    # read the yaml file
    try:
        with open(yaml_file, 'r') as f:
            yaml_content = f.read()
    except FileNotFoundError:
        raise ValueError(f"Config file '{yaml_file}' not found. Please check the path and try again.")

    # Parse YAML content
    try:
        parsed_yaml: dict[str, str | dict] = yaml.safe_load(yaml_content)
        return parsed_yaml
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {str(e)}")
    except ValueError as e:
        raise ValueError(f"Validation error: {str(e)}")
