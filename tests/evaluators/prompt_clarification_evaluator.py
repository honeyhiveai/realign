from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    prompt_clarification_evaluator:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                As an LLM prompt clarification evaluator, your task is to evaluate if the assistant appropriately seeks clarification when the user's intent is not clear. The clarification score should be between 0 and 1, where 0 means no clarification needed and 1 means clarification is highly needed. Respond with the clarification score in the following JSON format: {'clarification_score': GENERATED_CLARIFICATION_SCORE}. Here's the conversation: {{messages}}.
            json_mode: on
        target: '[0.7, 1]' # clarification score range
        in_range: numeric_range
"""

@evaluator
def prompt_clarification_evaluator(messages):
    '''Evaluates if the assistant appropriately seeks clarification when the user's intent is not clear.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the clarification score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['clarification_score']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_prompt_clarification_evaluator():
    Simulator(str_yaml=config, config_only=True)

    # adversarial test
    adversarial_state = [{'role': 'user', 'content': 'I need help with something.'}]
    evaluation = prompt_clarification_evaluator(adversarial_state).unpack()
    score, result = evaluation
    print(f"Adversarial Test - Score: {score}, Result: {result}")
    assert result == True

    # robustness test
    robust_state = [{'role': 'user', 'content': 'I need help with my math homework.'}]
    evaluation = prompt_clarification_evaluator(robust_state).unpack()
    score, result = evaluation
    print(f"Robustness Test - Score: {score}, Result: {result}")
    assert result == False

if __name__ == "__main__":
    test_prompt_clarification_evaluator()
