from typing import Any, Optional
from agentsim import AbstractAgent

class SimpleChatbot(AbstractAgent):
    def process_turn(self, messages: list) -> Optional[Any]:
        
        # Update state
        messages = self._generate_chat_completion(messages)
        
        # return the updated state
        return messages