from agentsim.utils import llm_eval_call, check_guardrail
from realign.evaluation import evaluator

@evaluator
def toxicity_score_llm(messages):
    
    # get the toxicity score
    score = llm_eval_call({
        'messages': str(messages)
    })['toxicity_score']
    
    # check if the score is in the target range
    result = check_guardrail(score)
    
    return score, result
