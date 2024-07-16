from dataclasses import dataclass
from typing import Any, Optional
import json
import hashlib
from jinja2 import Template
from  realign.prompts import RATING_5_STAR, SYNTH_USER_PROMPT_GENERATOR_TEMPLATE

def resolve_prompt_template(template_name: str):
    if template_name == 'rating_5_star':
        return RATING_5_STAR
    elif template_name == 'synthetic_user_prompt_generator':
        return SYNTH_USER_PROMPT_GENERATOR_TEMPLATE
    raise ValueError("Template not found")


@dataclass
class ModelSettings:
    model: str
    api_key: Optional[str] = None
    hyperparams: Optional[dict[str, Any]] = None
    prompt_params: Optional[dict[str, str]] = None
    template: Optional[str] = None
    system_prompt: Optional[str] = None
    json_mode: Optional[bool] = False
    role: str = 'assistant'
    
    def resolve_response_mode(self) -> str:
        if self.json_mode or self.template:
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
        prompt_params = self.prompt_params

        if prompt_params is None:
            return jinja_template.render({})
        elif type(prompt_params) != dict:
            raise ValueError("Prompt params must be a dictionary")
        elif not all([type(k) == str for k in prompt_params.keys()]):
            raise ValueError("Prompt params keys must be strings")
        
        # ensure that values are all strings
        for k, v in prompt_params.items():
            if type(k) != str:
                raise ValueError("Prompt params keys must be strings")
            if type(v) != str:
                prompt_params[k] = str(v)
        
        # try to render the template
        try:
            render = jinja_template.render(prompt_params)
        except Exception as e:
            raise ValueError(f"Error rendering system prompt: {e}")
        
        return render

@dataclass
class AgentConfig:
    architecture: Any
    model_settings: ModelSettings
    role: str


@dataclass
class AppConfig:
    agent: AgentConfig

@dataclass
class EvaluatorConfig:
    model_settings: Optional[ModelSettings]
    target: Optional[Any]
    in_range: Optional[str]
    
@dataclass
class EvaluatorsConfig:
    evaluators: dict[str, EvaluatorConfig]

@dataclass
class SimulationTestConfig:
    personas: dict[str, Any]
    scenarios: dict[str, Any]

@dataclass
class SimulationConfig:
    agent: AgentConfig
    tests: SimulationTestConfig

@dataclass
class AgentSimConfig:
    app: AppConfig
    evaluators: dict[str, EvaluatorConfig]
    robustness_simulations: SimulationConfig
    adversarial_simulations: SimulationConfig

@dataclass
class OpenAIMessage:
    role: str
    content: str | dict[str, str]

    def __dict__(self):
        return {
            'role': self.role,
            'content': str(self.content)
        }

@dataclass
class RunData:
    final_state: Any
    run_id: Optional[int] = None
    
    def __dict__(self):
        return {
            'run_id': self.run_id,
            'final_state': self.final_state
        }
    
    def __repr__(self) -> str:
        return str(self.__dict__())
    
    def compute_hash(self, hash_algorithm='sha256'):
        """
        Compute a hash of a RunData.
        
        :param obj: The object to hash
        :param hash_algorithm: The hash algorithm to use (default is 'sha256')
        :return: A hexadecimal string representation of the hash
        """
        # Convert the object to a JSON string
        json_string = json.dumps(self.__dict__(), sort_keys=True, default=str)
        
        # Create a hash object with the specified algorithm
        hash_object = hashlib.new(hash_algorithm)
        
        # Update the hash object with the JSON string (encoded to bytes)
        hash_object.update(json_string.encode('utf-8'))
        
        # Return the hexadecimal representation of the hash
        return hash_object.hexdigest()    

'''
---
app:
    agent_1:
        architecture: SimpleChatbot
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            hyperparams:
                temperature: 0.5
            system_prompt: >
                Be a good bot.
            json_mode: off
        role: 'assistant'
    
    agent_2: ...


evaluators:
    evaluator_1:
        model_settings:
            <<: *openai_gpt_4o
            template: rating_5_star
            system_prompt: Hello
            json_mode: on
            prompt_params: {}
        target: '[4,5]' # target score range
        in_range: numeric_range

    evaluator_2: on
    
    ... templated evaluators ...
    
adversarial_simulations:
    agent:
        architecture: SimpleChatbot
        model_settings: *mythomax
        role: 'user'
    
    tests:
        personas:
            persona_1: someone
            persona_2: someone
        scenarios:
            scenario_1: something
            scenario_2: something

    templates:
        explicit_violence: on
        sexual_content: on
        profanity: on
'''