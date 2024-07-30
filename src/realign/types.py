from realign.prompts import resolve_prompt_template
from dataclasses import dataclass
from litellm import validate_environment
from typing import Any, Optional
import json
import hashlib
from jinja2 import Template

@dataclass
class ModelSettings:
    # litellm model name. Refer to https://docs.litellm.ai/docs/providers.
    model: str 
    
    # API key env variable name. 
    # If not provided, defaults to <MODEL_PROVIDER>_API_KEY format
    api_key: Optional[str] = None
    
    # hyperparam dictionary in OpenAI format, eg. { 'temperature': 0.8 }
    hyperparams: Optional[dict[str, Any]] = None
		
    # literal system prompt
    # if provided, template/prompt_params will be ignored
    system_prompt: Optional[str] = None

    # Jinja template and prompt_param dictionary to render it
    # string key for the template. Actual templates defined in realign.prompts
    prompt_params: Optional[dict[str, str]] = None
    template: Optional[str] = None

    # json_mode for the response format
    json_mode: Optional[bool] = False
    
    # user or assistant
    role: str = 'assistant'
    
    def resolve_response_format(self) -> str:
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
    
    def validate_keys(self):
        # validate that the API keys are set
        model_key_validation = validate_environment(self.model)
        if not model_key_validation['keys_in_environment']:
            raise ValueError(f'Could not find the following API keys in the environment: {','.join(model_key_validation['missing_keys'])}. Please set these keys in the environment.')
    
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
        return f'eval_name:   {self.eval_name}\n' + \
               f'score:       {self.score}\n' + \
               f'result:      {self.result}\n' + \
               f'explanation: {self.explanation}\n'
    
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
    
    def to_dict(self) -> dict:
        return self.__dict__()
