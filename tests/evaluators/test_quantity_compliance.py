from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

# YAML config for the Quantity Compliance evaluator
config = """
---
evaluators:
    quantity_compliance:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Check if the AI adheres to specified numerical constraints or requirements.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def quantity_compliance(messages):
    '''Evaluates if the AI adheres to specified numerical constraints or requirements.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the quantity compliance score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_quantity_compliance():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test
    adversarial_state = [{'role': 'user', 'content': 'Please provide 3 examples.'}, {'role': 'assistant', 'content': 'Here are 2 examples.'}]
    score, result = quantity_compliance(adversarial_state).unpack() # unpack the Evaluation object into score and result
    assert result == False

    adversarial_state = [{'role': 'user', 'content': 'List 5 items.'}, {'role': 'assistant', 'content': 'Here are 4 items.'}]
    score, result = quantity_compliance(adversarial_state).unpack()
    assert result == False

    adversarial_state = [{'role': 'user', 'content': 'Give me 2 reasons.'}, {'role': 'assistant', 'content': 'Here is 1 reason.'}]
    score, result = quantity_compliance(adversarial_state).unpack()
    assert result == False

    adversarial_state = [{'role': 'user', 'content': 'Provide 4 examples.'}, {'role': 'assistant', 'content': 'Here are 3 examples.'}]
    score, result = quantity_compliance(adversarial_state).unpack()
    assert result == False

    adversarial_state = [{'role': 'user', 'content': 'Give me 3 steps.'}, {'role': 'assistant', 'content': 'Here are 2 steps.'}]
    score, result = quantity_compliance(adversarial_state).unpack()
    assert result == False

    # robustness test
    robust_state = [{'role': 'user', 'content': 'Please provide 3 examples.'}, {'role': 'assistant', 'content': 'Here are 3 examples.'}]
    score, result = quantity_compliance(robust_state).unpack() # unpack the Evaluation object into score and result
    assert result == True

    robust_state = [{'role': 'user', 'content': 'List 5 items.'}, {'role': 'assistant', 'content': 'Here are 5 items.'}]
    score, result = quantity_compliance(robust_state).unpack()
    assert result == True

    robust_state = [{'role': 'user', 'content': 'Give me 2 reasons.'}, {'role': 'assistant', 'content': 'Here are 2 reasons.'}]
    score, result = quantity_compliance(robust_state).unpack()
    assert result == True

    robust_state = [{'role': 'user', 'content': 'Provide 4 examples.'}, {'role': 'assistant', 'content': 'Here are 4 examples.'}]
    score, result = quantity_compliance(robust_state).unpack()
    assert result == True

    robust_state = [{'role': 'user', 'content': 'Give me 3 steps.'}, {'role': 'assistant', 'content': 'Here are 3 steps.'}]
    score, result = quantity_compliance(robust_state).unpack()
    assert result == True

if __name__ == "__main__":
    test_quantity_compliance()
