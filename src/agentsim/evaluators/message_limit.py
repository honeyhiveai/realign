from agentsim.utils import check_guardrail
from realign.evaluation import evaluator

@evaluator
def message_limit(messages):
    '''Returns the number of messages'''

    count = len(messages)
    result = check_guardrail(score=count)

    return count, result
