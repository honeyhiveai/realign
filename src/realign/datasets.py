import json

from realign.llm_utils import OpenAIMessage

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
