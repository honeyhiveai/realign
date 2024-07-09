from typing import Any, Optional
from agentsim import AbstractAgent
from agentsim.evaluators import message_limit

class SimpleChatbot(AbstractAgent):
    def process_turn(self, messages: list) -> Optional[Any]:

        # Add system prompt
        if len(messages) == 0:
            messages.append({ 'role': 'system', 'content': self.config.system_prompt })
        
        evaluation = message_limit(messages)
        print(f"Message count: {evaluation.score}")
        
        # Update state
        messages = self._generate_chat_completion(messages)
        return messages