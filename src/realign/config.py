import yaml
import os
from typing import Optional, Any
from dataclasses import dataclass

from jinja2 import Template
from litellm import validate_environment

import realign
from realign.prompts import resolve_prompt_template


# Module-level variable to store the config path
_config_path = 'src/realign/defaults.yaml'

DEFAULT_EVALUATOR_SETTINGS = {
    'weight': 1.0,
    'asserts': False,
    'repeat': None,
    'transform': None,
    'aggregate': None,
    'checker': None,
    'target': None
}

EVALUATOR_SETTINGS_KEYS = DEFAULT_EVALUATOR_SETTINGS.keys()

def get_config_path():
    global _config_path
    if _config_path is None:
        # If not set, try to get from environment variable
        _config_path = os.environ.get('REALIGN_CONFIG_PATH', _config_path)
    return _config_path

def set_config_path(path):
    global _config_path
    _config_path = path
    # Save to environment variable
    os.environ['REALIGN_CONFIG_PATH'] = path

# Create a property-like interface
config_path = property(get_config_path, set_config_path)

@dataclass
class ModelSettings:
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
    role: str = 'assistant'
    
    def resolve_response_format(self) -> str:
        if self.json_mode: 
            return { 'type': "json_object" }
        return None
    
    def resolve_system_prompt(self) -> str:
        prompt_to_render = ''
        system_prompt = self.system_prompt
        template = self.template
        if system_prompt == None:
            if template == None:
                raise ValueError("Either system_prompt or template must be provided in the model settings")
            else:
                prompt_to_render = resolve_prompt_template(template)
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
        
        # ensure that values are all strings
        for k, v in template_params.items():
            if type(k) != str:
                raise ValueError("Prompt params keys must be strings")
            if type(v) != str:
                template_params[k] = str(v)
        
        # try to render the template
        try:
            render = jinja_template.render(template_params)
        except Exception as e:
            raise ValueError(f"Error rendering system prompt: {e}")
        
        return render
    
    def validate_keys(self):
        # validate that the API keys are set
        model_key_validation = validate_environment(self.model)
        if not model_key_validation['keys_in_environment']:
            raise ValueError(f'Could not find the following API keys in the environment: {','.join(model_key_validation['missing_keys'])}. Please set these keys in the environment.')
    
    def copy(self) -> 'ModelSettings':
        return ModelSettings(
            model=self.model,
            api_key=self.api_key,
            hyperparams=self.hyperparams,
            template_params=self.template_params,
            template=self.template,
            system_prompt=self.system_prompt,
            json_mode=self.json_mode,
            role=self.role
        )

    def with_template_params(self, template_params: dict[str, str]) -> 'ModelSettings':
        self.template_params = template_params
        return self

@dataclass
class EvalSettings:
    type: str
    weight: float = DEFAULT_EVALUATOR_SETTINGS['weight']
    asserts: bool = DEFAULT_EVALUATOR_SETTINGS['asserts']
    repeat: Optional[int] = DEFAULT_EVALUATOR_SETTINGS['repeat']
    transform: Optional[str] = DEFAULT_EVALUATOR_SETTINGS['transform']
    aggregate: Optional[str] = DEFAULT_EVALUATOR_SETTINGS['aggregate']
    checker: Optional[str] = DEFAULT_EVALUATOR_SETTINGS['checker']
    target: Optional[str] = DEFAULT_EVALUATOR_SETTINGS['target']
    
    def copy(self) -> 'EvalSettings':
        return EvalSettings(
            type=self.type,
            weight=self.weight,
            repeat=self.repeat,
            asserts=self.asserts,
            transform=self.transform,
            aggregate=self.aggregate,
            checker=self.checker,
            target=self.target
        )
        
    def keys(self):
        return self.__dict__.keys()

def resolve_config_path(config_path) -> Optional[str]:
    if type(config_path) == str:
        return config_path
    elif type(config_path) == property:
        return config_path.fget()
    return None

def load_yaml_settings(yaml_file: Optional[str] = None) -> dict[str, dict]:
    """
    Load and parse a YAML configuration file.

    Args:
        yaml_file (Optional[str]): Path to the YAML file. If not provided, it will look for the
            REALIGN_CONFIG_PATH environment variable or the realign.config_path.

    Returns:
        dict[str, dict]: A dictionary containing the parsed YAML content.
    """
    
    # look for config.yaml in the current directory
    yaml_file = yaml_file or os.getenv('REALIGN_CONFIG_PATH') or resolve_config_path(realign.config_path)
    
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

def get_model_settings(yaml_file: Optional[str] = None,
                       agent_name: Optional[str] = None) -> dict[str, ModelSettings] | ModelSettings:
    
    parsed_yaml = load_yaml_settings(yaml_file)

    if not isinstance(parsed_yaml, dict) or 'llm_agents' not in parsed_yaml:
        raise ValueError("Invalid YAML structure. Expected 'llm_agents' key at the root level.")

    assert isinstance(parsed_yaml['llm_agents'], dict), "llm_agents must be a dictionary"

    model_settings = {}
    for _agent_name, settings in parsed_yaml['llm_agents'].items():
        model_settings[_agent_name] = ModelSettings(**settings)

    if agent_name is not None:
        if agent_name not in model_settings:
            raise ValueError(f"Agent '{agent_name}' not found in 'llm_agents' section.")
        return model_settings[agent_name]

    return model_settings

def extract_eval_settings_and_kwargs(settings: dict[str, Any]):

    eval_kwargs = {}
    eval_settings = {}
    
    for key, value in settings.items():
        if key in EVALUATOR_SETTINGS_KEYS:
            eval_settings[key] = value
        else:
            eval_kwargs[key] = value
    
    return eval_settings, eval_kwargs

def get_eval_settings(yaml_file: Optional[str] = None,
                      eval_type: Optional[str] = None) -> tuple[dict, dict]:
    
    parsed_yaml = load_yaml_settings(yaml_file)

    if not isinstance(parsed_yaml, dict) or 'evaluators' not in parsed_yaml:
        raise ValueError("Invalid YAML structure. Expected 'evaluators' key at the root level.")

    assert isinstance(parsed_yaml['evaluators'], dict), "evaluators must be a dictionary"

    evals_settings: dict[str, EvalSettings] = dict()
    evals_kwargs: dict[str, Any] = dict()
    for _eval_type, settings in parsed_yaml['evaluators'].items():
        eval_settings, eval_kwargs = extract_eval_settings_and_kwargs(settings)
        evals_settings[_eval_type] = EvalSettings(type=_eval_type, **eval_settings)
        evals_kwargs[_eval_type] = eval_kwargs

    if eval_type is not None:
        if eval_type not in evals_settings:
            return EvalSettings(type=eval_type, **DEFAULT_EVALUATOR_SETTINGS), dict()
        return evals_settings[eval_type], evals_kwargs[eval_type]

    # we should never get here
    raise ValueError("No eval_type provided")