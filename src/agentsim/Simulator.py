import yaml
from typing import Any, Optional
import importlib
import os
from jinja2 import Template
from dotenv import load_dotenv

from agentsim import AbstractAgent
from agentsim.types import *


class ModuleLoader:
    @staticmethod
    def load_class(package: str, module_name: str, load_module: bool = True):
        try:
            module = importlib.import_module('agentsim.' + package + '.' + module_name, package='agentsim')
            if load_module:
                if hasattr(module, module_name):
                    return getattr(module, module_name)
                else:
                    raise AttributeError(f"Module {module} has no class named {module_name}")
            else:
                return module
        except ImportError:
            raise ImportError(f"Could not import module {module_name} from package {package}")

class Simulator:
    def __init__(self,
                 yaml_file: str = None,
                 raw_yaml: dict[str, Any] = None,
                 str_yaml: str = None,
                 config_only: bool = False):
        self.raw_yaml = raw_yaml
        self.yaml_file = yaml_file
        self.str_yaml = str_yaml
        self.raw_config: dict[str, Any] = {}
        self.agentsim_config: Optional[AgentSimConfig] = None
        self.config_only: bool = config_only
        
        # app and sim agents
        self.app_agent: AbstractAgent = {}
        self.sim_agent: AbstractAgent = {}
        
        # Initialize the simulator
        self.initialize()
        
    @staticmethod
    def initialize_agent(agent_config: AgentConfig) -> AbstractAgent:
        agent_class = ModuleLoader.load_class('agents', agent_config.architecture)
        return agent_class(agent_config)

    def load_yaml(self):
        if self.yaml_file is not None:
            with open(self.yaml_file, 'r') as file:
                self.raw_config = yaml.safe_load(file)
        elif self.raw_yaml is not None:
            self.raw_config = yaml.load(self.raw_yaml, Loader=yaml.FullLoader)
        elif self.str_yaml is not None:
            self.raw_config = yaml.safe_load(self.str_yaml)
        else:
            raise ValueError("No config yaml_file or raw_yaml must be provided")

    def parse_model_settings(self, config: dict[str, Any]) -> ModelSettings:
        return ModelSettings(
            model=config.get('model', ''),
            api_key=config.get('api_key', ''),
            hyperparams=config.get('hyperparams', {}),
            system_prompt=config.get('system_prompt', ''),
            json_mode=config.get('json_mode', False)
        )

    def parse_agent_config(self, config: dict[str, Any]) -> AgentConfig:
        
        # override the system prompt from the agent config if provided
        model_settings = self.parse_model_settings(config.get('model_settings', {}))
        system_prompt = model_settings.system_prompt
        if config.get('system_prompt', '') != '':
            system_prompt = config.get('system_prompt')

        return AgentConfig(
            architecture=config.get('architecture', ''),
            model_settings=self.parse_model_settings(config.get('model_settings', {})),
            state=config.get('state', ''),
            system_prompt=system_prompt,
            role=config.get('role', 'assistant')
        )

    def load_app_config(self) -> AppConfig:
        app_config = self.raw_config.get('app', None)
        if not app_config: 
            return app_config

        agent_config = self.parse_agent_config(app_config.get('agent', {}))
        agent_config.role = 'assistant'

        # Load simulation modules
        if not self.config_only:
            self.app_agent = Simulator.initialize_agent(agent_config)

        return AppConfig(agent=agent_config)

    def load_evaluators_config(self) -> dict[str, EvaluatorConfig]:
        evaluators_config = self.raw_config.get('evaluators', {})
        if not evaluators_config: 
            return evaluators_config
        parsed_evaluators = {}
        
        for name, config in evaluators_config.items():
            config = config or {}
            parsed_evaluators[name] = EvaluatorConfig(
                model_settings=self.parse_model_settings(config.get('model_settings', {})),
                target=config.get('target', None),
                in_range=config.get('in_range', '')
            )

        return parsed_evaluators

    def load_simulations_config(self) -> SimulationConfig:
        simulations_config = self.raw_config.get('simulations', None)
        if not simulations_config:
            return simulations_config
        agent_config = self.parse_agent_config(simulations_config.get('agent', {}))
        agent_config.role = 'user'
        
        synth_user_settings = simulations_config.get('synth_user_settings', None)
        if not synth_user_settings:
            raise ValueError("Synthetic user settings must be provided in the simulations config")
        
        synth_user_settings = SyntheticUserSettings(personas=synth_user_settings.get('personas', {}),
                                                    scenarios=synth_user_settings.get('scenarios', {}),
                                                    model_settings=self.parse_model_settings(synth_user_settings.get('model_settings', {})),
                                                    shuffle_seed=synth_user_settings.get('shuffle_seed', None)
                                                    )
        
        # shuffle the personas and intents 
        if not self.config_only:
            if synth_user_settings.personas and synth_user_settings.scenarios:
                import random
                import json
                from litellm import completion
                
                if synth_user_settings.shuffle_seed:
                    random.seed(synth_user_settings.shuffle_seed)
                
                personas = list(synth_user_settings.personas.values())
                scenarios = list(synth_user_settings.scenarios.values())
                
                persona = random.choice(personas)
                scenario = random.choice(scenarios)
                
                synth_user_gen_prompt_template = Template(synth_user_settings.model_settings.system_prompt)
                response = completion(
                    model=synth_user_settings.model_settings.model,
                    api_key=os.getenv(synth_user_settings.model_settings.api_key),
                    messages=[
                        {
                            'role': 'user', 
                            'content': synth_user_gen_prompt_template.render({
                                'persona': persona, 
                                'scenario': scenario
                            })
                        },
                    ],
                    response_format={ 'type': "json_object" },
                    **synth_user_settings.model_settings.hyperparams
                )
                synth_user_system_prompt = json.loads(response.choices[0].message['content'])['synth_user_system_prompt']

                # set the system prompt for the synthetic user
                agent_config.system_prompt = synth_user_system_prompt
            else:
                raise ValueError("Personas and scenarios must be provided in the simulations config")

            # Load simulation modules
            self.sim_agent = Simulator.initialize_agent(agent_config)
        
        return SimulationConfig(
            agent=agent_config,
            synthetic_user=synth_user_settings
        )

    def initialize(self):
        self.load_yaml()
        
        # Load the modules if True
        app_config = self.load_app_config()
        evaluators_config = self.load_evaluators_config()
        simulations_config = self.load_simulations_config()
        
        self.agentsim_config = AgentSimConfig(
            app=app_config,
            evaluators=evaluators_config,
            simulations=simulations_config
        )
        
        # dump the json of the config to an environment variable
        os.environ['AGENTSIM_CONFIG'] = yaml.dump(self.raw_config)

        # load the environment variables
        load_dotenv()

    def get_raw_config(self) -> dict[str, Any]:
        return self.raw_config

    def get_agentsim_config(self) -> Optional[AgentSimConfig]:
        return self.agentsim_config

    def get_app_agent(self):
        return self.app_agent

    def get_simulation_app(self) -> dict[str, Any]:
        return self.sim_agent