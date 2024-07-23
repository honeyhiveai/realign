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
    input_template: Optional[str] = None
    
    def resolve_response_mode(self) -> str:
        if self.json_mode or self.template:
            return { 'type': "json_object" }
        return None
    
    def resolve_system_prompt(self, use_input_template=False) -> str:
        prompt_to_render = ''
        system_prompt = self.system_prompt
        template = self.template
        if system_prompt is None:
            if template is None:
                raise ValueError("Either system_prompt or template must be provided in the model settings")
            else:
                prompt_to_render = resolve_prompt_template(template)
        else:
            prompt_to_render = system_prompt
        
        if use_input_template:
            assert self.input_template is not None, "Input template must be provided to resolve prompt"

            response_format = '''Talk strictly in JSON format to populate the given template. Respond with a JSON where the keys are the template params for the template. Make sure that you respond with ALL the template params (marked with double curly braces) filled in. Do NOT include the curly braces in your response. Here is the template: \n[DERIVE TEMPLATE PARAM JSON FROM THIS TEMPLAET:]\n''' \
                + self.input_template

            self.json_mode = True
        else:
            response_format = '''Talk strictly in conversation format. Extremely important: ALL your responses should be ONE sentence only and no more. Start by introducing yourself and stating what you'd like to do.' followed by detailed instructions on how to proceed with the scenario.'''

        prompt_to_render += '\n\n' + response_format

        jinja_template = Template(prompt_to_render)
        prompt_params = self.prompt_params

        if prompt_params is None:
            return jinja_template.render({})
        elif type(prompt_params) != dict:
            raise ValueError("Prompt params must be a dictionary")
        elif not all([type(k) == str for k in prompt_params.keys()]):
            raise ValueError("Prompt params keys must be strings")
        
        # ensure that keys and values are all strings
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
    
    def copy(self):
        return ModelSettings(
            model=self.model,
            api_key=self.api_key,
            hyperparams=self.hyperparams,
            prompt_params=self.prompt_params,
            template=self.template,
            system_prompt=self.system_prompt,
            json_mode=self.json_mode,
            role=self.role
        )

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
    
# this object contains a single score
# the score can be of any type, including a list of scores
class EvalResult:
    def __init__(self, 
                 score: Any, result: bool | None, 
                 explanation: str | None = None, 
                 embedding = None,
                 run_data: RunData = None, 
                 eval_name: str | None = None, 
                 repeat: int = 1):
        self.score = score
        self.result = result
        self.explanation = explanation
        self.embedding = embedding
        self.run_data = run_data
        self.eval_name = eval_name
        self.repeat = repeat

    def __repr__(self):
        # get the object id of the run_data
        run_data_id = id(self.run_data) if self.run_data else None
        return f'(eval_name: {self.eval_name}, run_data: {run_data_id}, score: {self.score}, result: {self.result}, explanation: {self.explanation})'
    
    def __str__(self):
        return self.__repr__()
    
    def unpack(self):
        if self.explanation:
            return self.score, self.result, self.explanation
        return self.score, self.result
    
    def __dict__(self):
        return {
            self.eval_name: {
                'score': self.score,
                'result': self.result,
                'explanation': self.explanation,
            }
        }
    
    def to_dict(self):
        return self.__dict__()

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