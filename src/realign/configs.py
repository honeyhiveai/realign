import yaml
import os
from typing import Optional
import asyncio
import inspect

import realign


DEFAULT_CONFIG_PATH = "defaults.yaml"

# Export the config_path property for easy access
config_path = DEFAULT_CONFIG_PATH


class Config:

    _config_path = DEFAULT_CONFIG_PATH
    config_content = dict()

    def __set__(self, _, path):
        
        # get the path of the caller of this function using inspect
        caller_path = inspect.stack()[1].filename
        dir_path = os.path.join(os.path.dirname(caller_path), path)

        if os.path.exists(path):
            path = path
        elif os.path.exists(dir_path):
            path = dir_path
        else:
            raise FileNotFoundError(f"Config file {path} or {dir_path} not found.")

        # if path unchanged, return
        # if Config._config_path == path:
        #     return

        # check if valid yaml
        Config.config_content = dict()
        with open(path) as f:
            try:
                Config.config_content = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
                raise

            if not ("llm_agents" in Config.config_content or "evaluators" in Config.config_content):
                raise ValueError(
                    f"Invalid YAML structure. Expected 'llm_agents' or 'evaluators' keys at the root level."
                )

        Config._config_path = path
        self.load_config()

    def __get__(self, obj, objtype):
        return Config._config_path

    def __call__(self):
        self.load_config()
    
    @staticmethod
    def create_wrapper(base_callable, eval_name, coroutine):
        if coroutine:
            async def afunc(*args, **kwargs):
                return await base_callable(*args, **kwargs)
            afunc.__name__ = eval_name
            return afunc
        
        def func(*args, **kwargs):
            return base_callable(*args, **kwargs)
        func.__name__ = eval_name
        return func

    def load_config(self):

        # TODO: cleaner solution, circular import
        from realign.evaluators import evaluator, aevaluator, get_eval_settings

        config_eval_settings, config_eval_kwargs = get_eval_settings(
            yaml_file=Config._config_path
        )
        
        '''
        Logic to parse the config file.
        - define evaluator class
            - parse default config file
        - initialize all evallib evaluators using default config file
        
        - user code
            - if config.path is set, parse it and load its configs
            - when user defines evaluators, use config.path configs
        '''

        for eval_name, eval_settings in config_eval_settings.items():
            # create a wrapper eval if wraps is set
            if eval_settings.wraps is not None:
                # get the wrapped evaluator
                assert isinstance(eval_settings.wraps, str)
                base_callable = eval(eval_settings.wraps, evaluator.all_evaluators)
                assert isinstance(base_callable, evaluator)

                if asyncio.iscoroutinefunction(base_callable.func):
                    # make the wrapper eval
                    afunc = Config.create_wrapper(base_callable, 
                                                  eval_name, 
                                                  True)

                    # initialize the wrapper eval
                    aevaluator(
                        func=afunc,
                        eval_settings=config_eval_settings[eval_name],
                        eval_kwargs=config_eval_kwargs[eval_name],
                    )
                else:
                    # make the wrapper eval
                    func = Config.create_wrapper(base_callable,
                                                 eval_name,
                                                 False)

                    # initialize the wrapper eval
                    evaluator(
                        func=func,
                        eval_settings=config_eval_settings[eval_name],
                        eval_kwargs=config_eval_kwargs[eval_name],
                    )
            
            # update the existing evaluator
            # NOTE: this will override any args, kwargs, or deco_kwargs
            else:
                # update the settings based on the config
                if eval_name in evaluator.all_eval_settings:
                    evaluator.all_evaluators[eval_name].eval_settings.update(
                        config_eval_settings[eval_name]
                    )
                else:
                    evaluator.all_eval_settings[eval_name] = config_eval_settings[eval_name]

                # update the kwargs based on the config
                if eval_name in evaluator.all_eval_kwargs:
                    evaluator.all_evaluators[eval_name].eval_kwargs.update(
                        config_eval_kwargs[eval_name]
                    )
                else:
                    evaluator.all_eval_kwargs[eval_name] = config_eval_kwargs[eval_name]
                
                # update the evaluator.eval_settings and kwargs based on the config
                if eval_name in evaluator.all_evaluators:
                    evaluator.all_evaluators[eval_name].eval_settings.update(
                        config_eval_settings[eval_name]
                    )
                    evaluator.all_evaluators[eval_name].eval_kwargs.update(
                        config_eval_kwargs[eval_name]
                    )


class ConfigPath:
    path = Config()

    def __getitem__(self, key):
        return Config.config_content[key]

config = ConfigPath()


def load_yaml_settings(yaml_file: Optional[str] = None) -> dict[str, dict]:

    def resolve_config_path() -> Optional[str]:
        config_path = realign.config.path
        if type(config_path) == str:
            return config_path
        return None

    # look for config.yaml in the current directory
    yaml_file = yaml_file or resolve_config_path()

    if yaml_file is None:
        raise ValueError(
            "No config file specified. Please set the REALIGN_CONFIG_PATH environment variable or pass in a config file path."
        )

    # read the yaml file
    try:
        with open(yaml_file, "r") as f:
            yaml_content = f.read()
    except FileNotFoundError:
        # current directory / yaml file
        try:
            yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml_file)

            with open(yaml_file, "r") as f:
                yaml_content = f.read()
                
        except FileNotFoundError:
            
            raise ValueError(
                f"Config file '{yaml_file}' not found. Please check the path and try again."
            )

    # Parse YAML content
    try:
        parsed_yaml: dict[str, str | dict] = yaml.safe_load(yaml_content)

        return parsed_yaml
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {str(e)}")
    except ValueError as e:
        raise ValueError(f"Validation error: {str(e)}")
