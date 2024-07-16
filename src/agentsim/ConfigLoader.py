from typing import Optional, Any
from realign.types import AgentSimConfig, \
                            AppConfig, \
                            EvaluatorsConfig, \
                            RobustnessSimulationsConfig, \
                            AdversarialSimulationsConfig, \
                            AgentConfig, \
                            EvaluatorConfig, \
                            SimulationConfig, \
                            ModelSettings
                            
import yaml


class ConfigLoader:
    '''Class to compile a given YAML config and store the compiled version in the environment.'''
    
    ROOT_DIR = 'src/agentsim'
    
    def __init__(self, yaml_input: str | dict[str, Any] | None = None):
        
        if yaml_input is None:
            raise ValueError("No config provided. Please specify your YAML file path, YAML string, or a dict representing a YAML.")
        
        # YAML input
        self.yaml_input = yaml_input
        
        # dict version of YAML
        self.raw_config: dict[str, Any] = None
        
        # compiled config object
        self.agentsim_config: Optional[AgentSimConfig] = None

        # Initialize the simulator
        self.initialize()
        
    def initialize(self):

        # Load the imports and the YAML file
        self.raw_config = ConfigLoader.load_yaml(self.yaml_input, [
            'models.yaml', 
            'evaluators.yaml',
            'agents.yaml'
        ])
        
        
    @staticmethod
    def load_yaml(yaml_input: str | dict[str, Any] | None, imports: list[str] = None):
        '''Load the YAML file into a dictionary.'''
        
        imported_yaml = '---\n'
        if imports:
            for file in imports:
                with open(ConfigLoader.ROOT_DIR + '/configs/' + file, 'r') as file:
                    imported_yaml += file.read()

        raw_config = imported_yaml

        # Load the YAML file into raw_config
        if '.yaml' in yaml_input:
            with open(yaml_input, 'r') as config_file:
                raw_config += config_file.read()
            raw_config = yaml.load(raw_config, Loader=yaml.FullLoader)
        elif type(yaml_input) == str:
            raw_config = yaml.safe_load(raw_config + yaml_input)
        else:
            raise ValueError("Could not parse the given YAML input. Please specify your YAML file path, YAML string, or a dict representing a YAML.")

        return raw_config
    
    def parse_app_config(self, app_config: dict[str, Any] | None) -> AppConfig:
        '''Load the app config from the raw config'''

        if app_config is None:
            raise ValueError("App config must be provided in the config file.")

        agent_config = self.parse_agent_config(app_config.get('agent', None))
        agent_config.role = 'assistant'

        return AppConfig(agent=agent_config)

    def parse_evaluator_config(self, evaluator_config: dict[str, Any]) -> dict[str, Any]:
        '''Load the evaluators config from the raw config'''

        # evaluators_config: dict[str, Any] = self.raw_config.get('evaluators', {})
        # if not evaluators_config: 
        #     return evaluators_config


        parsed_evaluators = {}
        for name, config in evaluator_config.items():
            parsed_evaluators[name] = EvaluatorConfig(
                model_settings=self.parse_model_settings(config.get('model_settings', {})),
                target=config.get('target', None),
                in_range=config.get('in_range', 'numeric_range')
            )
            

    def parse_model_settings(self, model_settings_config: dict[str, Any]) -> ModelSettings:
        '''Parse the model settings from the config.'''
        
        default_json_mode = model_settings_config.get('template', None) is not None

        return ModelSettings(
            model=model_settings_config.get('model', ''),
            api_key=model_settings_config.get('api_key', ''),
            hyperparams=model_settings_config.get('hyperparams', {}),
            system_prompt=model_settings_config.get('system_prompt', ''),
            template=model_settings_config.get('template', ''),
            prompt_params=model_settings_config.get('prompt_params', {}),
            # json mode is on by default if template is provided
            json_mode=model_settings_config.get('json_mode', default_json_mode)
        )

    def parse_agent_config(self, agent_config: str | dict[str, Any] | None) -> AgentConfig:
        
        if not agent_config:
            raise ValueError("Agent config must be provided in the config file.")
        
        return AgentConfig(
            architecture=agent_config.get('architecture', ''),
            model_settings=self.parse_model_settings(agent_config.get('model_settings', {})),
            role=agent_config.get('role', 'assistant')
        )


            
if __name__ == '__main__':
    config = ConfigLoader('src/agentsim/config.yaml')
    print(config.raw_config)