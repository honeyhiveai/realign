import yaml
from typing import Any, Optional
import importlib
import os
from dotenv import load_dotenv

from agentsim.types.types import *

from litellm import completion  
from agentsim.types.types import AgentConfig
from agentsim.utils import resolve_system_prompt

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
        self.sim_agent: AbstractAgent = {}
        self.app_agent: AbstractAgent = {}
        self.evaluators: dict[str, callable] = {}
        
        # Initialize the simulator
        self.initialize()
        
    @staticmethod
    def initialize_agent(agent_config: AgentConfig) -> AbstractAgent:
        agent_class = ModuleLoader.load_class('agents', agent_config.architecture)
        return agent_class(agent_config)


            # load the templates
            # if name == 'templates':
                # load the yaml from evaluators/default_evaluators.yaml
                # default_evaluators = {}
                # try:
                #     with open('evaluators/default_evaluators.yaml', 'r') as file:
                #         default_evaluators = yaml.safe_load(file)
                # except FileNotFoundError:
                #     raise FileNotFoundError("Could not find default evaluators file. Please make sure it exists.")

                # for config_template_eval, on_off in config.items():
                #     if type(on_off) != bool:
                #             raise ValueError(f"Template config must be a boolean values. Please use on or off, or True or False.")
                #     if on_off:
                #         retrieved_eval_config = default_evaluators.get(config_template_eval, None)
                #         if retrieved_eval_config is None:
                #             raise ValueError(f"Could not find template {config_template_eval}. Please make sure the evaluator is one of the defaults.")
                        
                #         # set the config to the retrieved config
                #         config = retrieved_eval_config


        
        # load the evaluators if not config only
        # if not self.config_only:
        #     for name, config in parsed_evaluators.items():
        #         self.evaluators[name] = ModuleLoader.load_class('evaluators', name)

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
                # import random
                # import json
                # from litellm import completion
                
                # if synth_user_settings.shuffle_seed:
                #     random.seed(synth_user_settings.shuffle_seed)
                
                # personas = list(synth_user_settings.personas.values())
                # scenarios = list(synth_user_settings.scenarios.values())
                
                # persona = random.choice(personas)
                # scenario = random.choice(scenarios)
                
                # synth_user_gen_prompt_template = Template(synth_user_settings.model_settings.system_prompt)

                # response = completion(
                #     model=synth_user_settings.model_settings.model,
                #     api_key=os.getenv(synth_user_settings.model_settings.api_key),
                #     messages=[
                #         {
                #             'role': 'user', 
                #             'content': synth_user_gen_prompt_template.render({
                #                 'persona': persona, 
                #                 'scenario': scenario
                #             })
                #         },
                #     ],
                #     response_format={ 'type': "json_object" },
                #     **synth_user_settings.model_settings.hyperparams
                # )
                # synth_user_system_prompt = json.loads(response.choices[0].message['content'])['synth_user_system_prompt']
                
                # set the system prompt for the synthetic user
                # agent_config.system_prompt = synth_user_system_prompt
                agent_config.system_prompt = synth_user_settings.model_settings.system_prompt
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