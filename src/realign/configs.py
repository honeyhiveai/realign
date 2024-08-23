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
    def __init__(self):
        self._config_path = DEFAULT_CONFIG_PATH

    def __set__(self, _, path):
        
        # get the path of the caller of this function using inspect
        caller_path = inspect.stack()[1].filename
        
        dir_path = os.path.join(os.path.dirname(caller_path), 'config.yaml')

        if os.path.exists(path):
            path = path
        elif os.path.exists(dir_path):
            path = dir_path
        else:
            raise FileNotFoundError(f"Config file not found at {path} or {dir_path}.")

        # if path unchanged, return
        # if self._config_path == path:
        #     return

        # check if valid yaml
        config_content = None
        with open(path) as f:
            try:
                config_content = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

            if not ("llm_agents" in config_content or "evaluators" in config_content):
                raise ValueError(
                    f"Invalid YAML structure. Expected 'llm_agents' or 'evaluators' keys at the root level."
                )

        self._config_path = path
        self.load_config()

    def __get__(self, obj, objtype):
        return self._config_path

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

        all_eval_settings, all_eval_kwargs = get_eval_settings(
            yaml_file=self._config_path
        )

        for eval_name, eval_settings in all_eval_settings.items():
            # create a wrapper eval if wraps is set
            if eval_settings.wraps is not None:
                # get the wrapped evaluator
                assert isinstance(eval_settings.wraps, str)
                base_callable = eval(eval_settings.wraps, evaluator.all_evaluators)
                assert isinstance(base_callable, evaluator)

                if asyncio.iscoroutinefunction(base_callable.func):

                    afunc = Config.create_wrapper(base_callable, 
                                                  eval_name, 
                                                  True)

                    # make the wrapper eval
                    aevaluator(
                        func=afunc,
                        eval_settings=all_eval_settings[eval_name],
                        eval_kwargs=all_eval_kwargs[eval_name],
                    )
                else:
                    func = Config.create_wrapper(base_callable, 
                                                  eval_name, 
                                                  False)

                    # make the wrapper eval
                    evaluator(
                        func=func,
                        eval_settings=all_eval_settings[eval_name],
                        eval_kwargs=all_eval_kwargs[eval_name],
                    )
            
            # update the existing evaluator
            else:

                # update their settings based on the config
                for eval_name in evaluator.all_evaluators.keys():
                    # update the settings
                    if eval_name in all_eval_settings:
                        if eval_name not in evaluator.all_eval_settings:
                            evaluator.all_eval_settings[eval_name] = all_eval_settings[eval_name]
                            
                        evaluator.all_eval_settings[eval_name].update(all_eval_settings[eval_name])
                        evaluator.all_evaluators[eval_name].eval_settings.update(
                            all_eval_settings[eval_name]
                        )

                    # update the kwargs
                    if eval_name in all_eval_kwargs:
                        if eval_name not in evaluator.all_eval_kwargs:
                            evaluator.all_eval_kwargs[eval_name] = all_eval_kwargs[eval_name]
                        
                        evaluator.all_eval_kwargs[eval_name].update(all_eval_kwargs[eval_name])
                        evaluator.all_evaluators[eval_name].eval_kwargs.update(
                            all_eval_kwargs[eval_name]
                        )

        # update the config path
        print("Loaded config file:", self._config_path)


class ConfigPath:
    path = Config()


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
