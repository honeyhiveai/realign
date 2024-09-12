import yaml
import os
import sys
from typing import Optional, Any
import inspect
import warnings

from .evaluators import evaluator, EvalSettings, EvaluatorSettings
from .llm_utils import all_agent_settings, all_agent_tools, AgentSettings
from .utils import bcolors, dotdict


DEFAULT_CONFIG_PATH = "defaults.yaml"
USER_CONFIG_PATH = "config.yaml"


def assume_config() -> Optional[str]:
    try:
        # Get the directory containing the file being run
        if getattr(sys, 'frozen', False):
            # If the application is run as a bundle, use the sys._MEIPASS
            running_dir = sys._MEIPASS
        else:
            # If it's not bundled, use the directory containing the script
            running_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

        # Prepend the directory containing the running file to config.path
        config_path = os.path.join(running_dir, USER_CONFIG_PATH)
        if os.path.exists(config_path):
            return config_path
        

    except Exception as e:
        print(f'Error while loading config file: {e}')
        print('Continuing...')
        
    return None


def load_yaml(_yaml) -> dict:
    if _yaml is None:
        return None
    
    content = None
    try:
        content = yaml.safe_load(_yaml)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {str(e)}")
    except ValueError as e:
        raise ValueError(f"Validation error: {str(e)}")
        
    return content

class Config:

    config_paths = []
    config_contents = []
    
    @staticmethod
    def resolve_path(path: str) -> str:
        # get the path of the caller of this function using inspect
        caller_path = inspect.stack()[1].filename
        dir_path = os.path.join(os.path.dirname(caller_path), path)
        
        # maybe try this?
        # os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml_file)

        if os.path.exists(path):
            path = path
        elif os.path.exists(dir_path):
            path = dir_path
        else:
            raise FileNotFoundError(f"Config file {path} or {dir_path} not found. Please check the path and try again.")     
        
        return path
    
    @staticmethod
    def get_yaml_content(path: str | None = None) -> dotdict:
        if path is None:
            raise ValueError("Please specify a config file path.")
        
        assert os.path.exists(path), f"Config file {path} not found. Please check the path and try again."
        with open(path) as f:
            content = load_yaml(f)
        
        content = content or dict()
        
        return dotdict(content)

    def __set__(self, _, path: Optional[str | Any] = None):   
        
        if path is None:
            return
        
        # check if path is a file or directory
        if '.yaml' in path or '.yml' in path: 
        
            # resolve and update the paths
            resolved_path = Config.resolve_path(path)
            
            # resolve and update the yaml contents
            resolved_yaml_content = Config.get_yaml_content(resolved_path)
            
            print(bcolors.OKBLUE, "Parsed config file:", resolved_path, bcolors.ENDC)
            
        elif isinstance(path, str):
            resolved_path = inspect.stack()[1].filename
            resolved_yaml_content = load_yaml(path)
            print(bcolors.OKBLUE, "Parsed config string in file:", os.path.basename(resolved_path), bcolors.ENDC)
        
        else:
            raise ValueError("Invalid config path. Please specify a file/directory or config string.")
        
        # append the path and content to the config
        Config.config_paths.append(resolved_path)
        Config.config_contents.append(resolved_yaml_content)  # This is now a dotdict
            
        # initialize the evaluators and llm agents
        Config.initialize_evaluators(resolved_yaml_content, resolved_path)
        Config.initialize_llm_agents(resolved_yaml_content, resolved_path)
        Config.initialize_tools(resolved_yaml_content, resolved_path)
        
    def __get__(self, obj, objtype):
        if len(Config.config_paths) == 0:
            raise ValueError("No config file loaded.")
        return Config.config_paths[-1]

    @staticmethod
    def initialize_evaluators(config_contents: dict, config_path: str | None = None):
        
        assert isinstance(config_contents, dict), f"Invalid YAML structure: {config_contents}"
        
        if "evaluators" not in config_contents:
            return
        
        assert isinstance(
            config_contents["evaluators"], dict
        ), "evaluators must be a dictionary"
        
        # parse evaluators in the config file
        for name, raw_settings in config_contents["evaluators"].items():
            eval_settings, eval_kwargs = EvalSettings.extract_eval_settings_and_kwargs(raw_settings)
            
            if name not in evaluator.all_evaluator_settings:
                evaluator.all_evaluator_settings[name] = EvaluatorSettings(name=name)
            
            if config_path and DEFAULT_CONFIG_PATH in config_path:
                evaluator.all_evaluator_settings[name].defaults_yaml_settings.update(eval_settings)
                evaluator.all_evaluator_settings[name].defaults_yaml_kwargs.update(eval_kwargs)
            else:
                evaluator.all_evaluator_settings[name].config_settings.update(eval_settings)
                evaluator.all_evaluator_settings[name].config_kwargs.update(eval_kwargs)

    @staticmethod
    def initialize_llm_agents(config_contents: dict, config_path: str | None = None):
        
        assert isinstance(config_contents, dict), f"Invalid YAML structure: {config_contents}"
        
        if "llm_agents" not in config_contents:
            return

        assert isinstance(
            config_contents["llm_agents"], dict
        ), "llm_agents must be a dictionary"

        for name, raw_settings in config_contents["llm_agents"].items():
            if name not in all_agent_settings:
                all_agent_settings[name] = AgentSettings(**raw_settings)
            
            if config_path and DEFAULT_CONFIG_PATH in config_path:
                all_agent_settings[name].update(AgentSettings(**raw_settings))
            else:
                all_agent_settings[name].update(AgentSettings(**raw_settings))

    @staticmethod
    def initialize_tools(config_contents: dict, config_path: str | None = None):
        assert isinstance(config_contents, dict), f"Invalid YAML structure: {config_contents}"
        
        if "tools" not in config_contents:
            return
        
        assert isinstance(
            config_contents["tools"], dict
        ), "tools must be a dictionary"
        
        for name, raw_settings in config_contents["tools"].items():
            all_agent_tools[name] = raw_settings
                
        
class ConfigPath:
    yaml = path = Config()
    
    @property
    def default(self):
        return Config.config_contents[0]
    
    def __getattr__(self, key):
        return Config.config_contents[-1][key]
        
    def __getitem__(self, key):
        return Config.config_contents[-1][key]
    
    def __setitem__(self, key, value):
        raise NotImplementedError("ConfigPath is read-only")

config = ConfigPath()