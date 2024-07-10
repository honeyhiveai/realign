from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    code_reusability:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the code snippet provided below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the code performs relative to the criteria. Be as objective as possible. After providing your explanation, you must rate the code on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}', for example: '{"rating": 2}'.
                [Criteria]

                Rate the reusability of the given code snippet, with 5 being highly reusable and 1 being not reusable at all.

                [The Start of Input]
                {{code_snippet}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def code_reusability(code_snippet):
    '''Evaluates the reusability of a code snippet by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'code_snippet': str(code_snippet)
    }

    # get the reusability score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_code_reusability():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test
    adversarial_state = [{'role': 'user', 'content': 'def add(a, b): return a + b'}]
    score, result = code_reusability(adversarial_state).unpack() # unpack the Evaluation object into score and result
    assert result == False

    adversarial_state_2 = [{'role': 'user', 'content': 'def subtract(a, b): return a - b'}]
    score, result = code_reusability(adversarial_state_2).unpack()
    assert result == False

    adversarial_state_3 = [{'role': 'user', 'content': 'def multiply(a, b): return a * b'}]
    score, result = code_reusability(adversarial_state_3).unpack()
    assert result == False

    adversarial_state_4 = [{'role': 'user', 'content': 'def divide(a, b): return a / b'}]
    score, result = code_reusability(adversarial_state_4).unpack()
    assert result == False

    adversarial_state_5 = [{'role': 'user', 'content': 'def modulus(a, b): return a % b'}]
    score, result = code_reusability(adversarial_state_5).unpack()
    assert result == False

    # robustness test
    robust_state = [{'role': 'user', 'content': 'def add(a, b):\n    """Adds two numbers and returns the result."""\n    return a + b'}]
    score, result = code_reusability(robust_state).unpack() # unpack the Evaluation object into score and result
    assert result == True

    robust_state_2 = [{'role': 'user', 'content': 'def subtract(a, b):\n    """Subtracts the second number from the first and returns the result."""\n    return a - b'}]
    score, result = code_reusability(robust_state_2).unpack()
    assert result == True

    robust_state_3 = [{'role': 'user', 'content': 'def multiply(a, b):\n    """Multiplies two numbers and returns the result."""\n    return a * b'}]
    score, result = code_reusability(robust_state_3).unpack()
    assert result == True

    robust_state_4 = [{'role': 'user', 'content': 'def divide(a, b):\n    """Divides the first number by the second and returns the result."""\n    return a / b'}]
    score, result = code_reusability(robust_state_4).unpack()
    assert result == True

    robust_state_5 = [{'role': 'user', 'content': 'def modulus(a, b):\n    """Returns the modulus of the first number by the second."""\n    return a % b'}]
    score, result = code_reusability(robust_state_5).unpack()
    assert result == True

if __name__ == "__main__":
    test_code_reusability()
