from abc import ABC, abstractmethod
from typing import Any, Optional
from litellm import completion  
from agentsim.types import AgentConfig
import os

class AbstractAgent(ABC):
    def __init__(self, config: AgentConfig):
        self.config: AgentConfig = config

    def _generate_chat_completion(self, messages: Any) -> str:
        
        # check if messages are empty
        if len(messages) == 0:
            raise ValueError('Messages cannot be empty')
        
        # swap roles for user
        if self.config.role == 'user':
            messages = AbstractAgent.swap_roles(messages)
        
        # set the system prompt assuming it is the first message
        messages[0] = { 'role': 'system', 'content': self.config.system_prompt }

        # make the completion call
        response = completion(
            model=self.config.model_settings.model,
            api_key=os.getenv(self.config.model_settings.api_key),
            messages=messages,
            **self.config.model_settings.hyperparams
        )
        
        # undo swap roles for user
        if self.config.role == 'user':
            messages = AbstractAgent.swap_roles(messages)

        # get response and add the right role
        response = response.choices[0].message
        response['role'] = self.config.role
        messages.append(response)

        return messages

    @staticmethod
    def swap_roles(messages: Any) -> Any:
        for message in messages:
            if message['role'] == 'user':
                message['role'] = 'assistant'
            elif message['role'] == 'assistant':
                message['role'] = 'user'
        return messages

    @abstractmethod
    def process_turn(self, state) -> Optional[Any]:
        """
        Process a turn based on the current state.
        
        :param state: The current state of the conversation or task.
        :return: A new state if processing should continue, or None if it should stop.
        """
        pass
