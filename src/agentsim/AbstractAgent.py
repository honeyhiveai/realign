from abc import ABC, abstractmethod
from typing import Any, Optional
from litellm import completion  
from realign.types import AgentConfig
import os

class AbstractAgent(ABC):
    def __init__(self, config: AgentConfig):
        self.config: AgentConfig = config

    def _generate_chat_completion(self, messages: list[dict[str, str]]) -> str:

        # swap roles for user
        if self.config.role == 'user':
            messages = AbstractAgent.swap_roles(messages)
        
        # set the system prompt assuming it is the first message
        messages[0] = { 'role': 'system', 'content': resolve_system_prompt(self.config.model_settings) }

        # make the completion call
        response = completion(
            model=self.config.model_settings.model,
            api_key=os.getenv(self.config.model_settings.api_key),
            messages=messages,
            **self.config.model_settings.hyperparams
        )
        
        # unswap roles for user
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
