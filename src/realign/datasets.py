from realign.types import OpenAIMessage
import json
from typing import Self
from realign.agents import SyntheticUserBuilder, ChatAgent

class Dataset:
    # TODO: async for to validate and load large datasets

    @staticmethod
    def validate_data_format(data) -> bool:
        if not data:
            raise ValueError("No data found in the dataset")

        # data must be a dictionary
        if type(data) != dict:
            raise ValueError("Dataset must be a dictionary")

        # data must have inputs, outputs, gtound_truth and metadata keys
        for key in ['inputs', 'outputs', 'ground_truths', 'metadata']:
            if key not in data:
                raise ValueError(f"Dataset must have a '{key}' key")
        
        return True

    def __init__(self, file_path: str = None):
        if not file_path:
            self.data = {
                'inputs': [],
                'outputs': [],
                'ground_truths': [],
                'metadata': []
            }
            return

        self.data = None
        if '.json' not in file_path:
            raise ValueError("Dataset file must be a json")

        if '.json' in file_path:
            with open(file_path) as f:
                data = json.load(f)
                if Dataset.validate_data_format(data):
                    self.data = data

class ChatDataset(Dataset):
 
    def validate_and_load_chat(self) -> bool:

        # load each messages in the ground truth
        for i in range(len(self.data['outputs'])):

            # output must be a dictionary
            if type(self.data['outputs'][i]) != dict:
                raise ValueError("Output must be a dictionary")
            
            # ground truth must have messages key
            if 'messages' not in self.data['outputs'][i]:
                raise ValueError("Outputs must have a 'messages' key")

            messages = []
            for message in self.data['outputs'][i]['messages']:
                if 'role' not in message or 'content' not in message:
                    raise ValueError("Each message in the ground truth must have a 'role' and 'content' key")
                messages.append(OpenAIMessage(role=message['role'], content=message['content']))
            self.data['outputs'][i]['messages'] = messages
        return True
 
    def __init__(self, file_path: str):
        super().__init__(file_path) # sets self.data
        self.validate_and_load_chat() # validates and loads the chat data into self.data
        
        self.synth_user_builder = SyntheticUserBuilder()
        
        self.synth_users = []
        
    def for_app(self, app: ChatAgent) -> Self:
        self.app = app
        self.synth_user_builder.synth_user_builder_model_settings.prompt_params['app'] = app.model_settings.resolve_system_prompt()
        return self

    def with_seed(self, persona, scenario) -> Self:
        self.synth_user_builder.as_a(persona).they_want_to(scenario)
        return self
    
    def generate_inputs(self, input_template: str, num_inputs: int) -> Self:
        # generate inputs
        self.synth_user_builder.with_num_personas(num_inputs)
        
        self.synth_user_builder.fetch_personas()
        
        assert self.synth_user_builder.retrieved_personas, "Personas must be fetched"
        
        self.synth_user_builder.with_input_template(input_template)
        
        for _ in range(num_inputs):
            synth_user_agent = self.synth_user_builder.build()
            self.synth_users.append(synth_user_agent)

            self.data['inputs'].append(synth_user_agent.generate_input())
        return self