from agentsim import Simulator
from agentsim.evaluators import message_limit
from agentsim.utils import check_guardrail

config="""
---
evaluators:
    message_limit:
        target: '(, 3]'
        in_range: numeric_range
""" 

def test_message_limit():
    Simulator(str_yaml=config, config_only=True)

    messages = []
    
    for i in range(10):
        e = message_limit(messages)
        assert e.score == i
        assert e.result == check_guardrail(i, 'message_limit')
        assert e.eval_name == 'message_limit'
        messages.append([{'role': 'user', 'content': 'Hello'}])

if __name__ == '__main__':
    test_message_limit()