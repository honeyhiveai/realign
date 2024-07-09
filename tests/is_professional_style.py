from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

@evaluator
def is_professional_style(messages):
    
    score = llm_eval_call({'messages': messages})['professionalism_score']
    
    result = check_guardrail(score)
    
    return score, result