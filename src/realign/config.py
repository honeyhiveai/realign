import yaml
import os
from typing import Optional

import realign

class ConfigPathManager:
    def __init__(self):
        self._config_path = 'src/realign/defaults.yaml'

    def get_config_path(self):
        if self._config_path is None:
            # If not set, try to get from environment variable
            self._config_path = os.environ.get('REALIGN_CONFIG_PATH', self._config_path)
        return self._config_path

    def set_config_path(self, path):
        self._config_path = path
        # Save to environment variable
        os.environ['REALIGN_CONFIG_PATH'] = path

    config_path = property(get_config_path, set_config_path)

# Create a single instance of the manager
config_manager = ConfigPathManager()

# Export the config_path property for easy access
config_path = config_manager.config_path


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
