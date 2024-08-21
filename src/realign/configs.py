import yaml
import os
from typing import Optional
import asyncio

import realign


DEFAULT_CONFIG_PATH = "src/realign/defaults.yaml"

# Export the config_path property for easy access
config_path = DEFAULT_CONFIG_PATH


class Config:
    def __init__(self):
        self._config_path = DEFAULT_CONFIG_PATH

    def __set__(self, _, path):

        print(f"Loading config from {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found at {path}")

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

    def load_config(self):

        # TODO: cleaner solution, circular import
        from realign.evaluators import evaluator, aevaluator, get_eval_settings

        all_eval_settings, all_eval_kwargs = get_eval_settings(
            yaml_file=self._config_path
        )

        evaluator.all_eval_settings.update(all_eval_settings)
        evaluator.all_eval_kwargs.update(all_eval_kwargs)

        for eval_name, eval_settings in all_eval_settings.items():
            if eval_settings.wraps is not None:
                # get the wrapped evaluator
                assert isinstance(eval_settings.wraps, str)
                base_callable = eval(eval_settings.wraps, evaluator.all_evaluators)
                assert isinstance(base_callable, evaluator)

                if asyncio.iscoroutinefunction(base_callable.func):

                    async def afunc(*args, **kwargs):
                        return await base_callable(*args, **kwargs)

                    afunc.__name__ = eval_name

                    # update the name of the wrapper evaluator
                    wrapper_eval = aevaluator(
                        func=afunc,
                        eval_settings=all_eval_settings[eval_name],
                        eval_kwargs=all_eval_kwargs[eval_name],
                    )
                else:

                    def func(*args, **kwargs):
                        return base_callable(*args, **kwargs)

                    func.__name__ = eval_name

                    # update the name of the wrapper evaluator
                    wrapper_eval = evaluator(
                        func=func,
                        eval_settings=all_eval_settings[eval_name],
                        eval_kwargs=all_eval_kwargs[eval_name],
                    )

                # create a new evaluator with the wraps callable
                evaluator.all_evaluators[eval_name] = wrapper_eval

        # update their settings based on the config
        for eval_name in evaluator.all_evaluators.keys():
            if eval_name in all_eval_settings and isinstance(
                all_eval_settings[eval_name], evaluator
            ):
                evaluator.all_evaluators[eval_name].eval_settings.update(
                    all_eval_settings[eval_name]
                )

            if eval_name in all_eval_kwargs and isinstance(
                all_eval_settings[eval_name], evaluator
            ):
                evaluator.all_evaluators[eval_name].eval_kwargs.update(
                    all_eval_kwargs[eval_name]
                )

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
