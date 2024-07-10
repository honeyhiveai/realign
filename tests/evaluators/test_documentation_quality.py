from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    documentation_quality:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality and completeness of the code documentation or comments in the provided code snippet. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the documentation performs relative to the code. Be as objective as possible. After providing your explanation, you must rate the documentation on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}', for example: '{"rating": 2}'.
                [Criteria]

                Rate the quality and completeness of the documentation in the given code snippet, with 5 being the highest quality and 1 being the lowest quality.

                [The Start of Input]
                {{code_snippet}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def documentation_quality(code_snippet):
    '''Evaluates the quality and completeness of code documentation or comments by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'code_snippet': str(code_snippet)
    }

    # get the documentation quality score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_documentation_quality():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test
    adversarial_state = [{'role': 'user', 'content': 'def foo():\n    pass\n# This function does nothing'}]
    score, result = documentation_quality(adversarial_state).unpack() # unpack the Evaluation object into score and result
    assert result == False

    adversarial_state_2 = [{'role': 'user', 'content': 'def bar():\n    pass\n# This function is a placeholder'}]
    score, result = documentation_quality(adversarial_state_2).unpack()
    assert result == False

    adversarial_state_3 = [{'role': 'user', 'content': 'def baz():\n    pass\n# No documentation provided'}]
    score, result = documentation_quality(adversarial_state_3).unpack()
    assert result == False

    adversarial_state_4 = [{'role': 'user', 'content': 'def qux():\n    pass\n# Incomplete documentation'}]
    score, result = documentation_quality(adversarial_state_4).unpack()
    assert result == False

    adversarial_state_5 = [{'role': 'user', 'content': 'def quux():\n    pass\n# Poorly written documentation'}]
    score, result = documentation_quality(adversarial_state_5).unpack()
    assert result == False

    # robustness test
    robust_state = [{'role': 'user', 'content': 'def add(a, b):\n    """Adds two numbers and returns the result."""\n    return a + b'}]
    score, result = documentation_quality(robust_state).unpack() # unpack the Evaluation object into score and result
    assert result == True

    robust_state_2 = [{'role': 'user', 'content': 'def subtract(a, b):\n    """Subtracts the second number from the first and returns the result."""\n    return a - b'}]
    score, result = documentation_quality(robust_state_2).unpack()
    assert result == True

    robust_state_3 = [{'role': 'user', 'content': 'def multiply(a, b):\n    """Multiplies two numbers and returns the result."""\n    return a * b'}]
    score, result = documentation_quality(robust_state_3).unpack()
    assert result == True

    robust_state_4 = [{'role': 'user', 'content': 'def divide(a, b):\n    """Divides the first number by the second and returns the result."""\n    return a / b'}]
    score, result = documentation_quality(robust_state_4).unpack()
    assert result == True

    robust_state_5 = [{'role': 'user', 'content': 'def modulus(a, b):\n    """Returns the modulus of the first number by the second."""\n    return a % b'}]
    score, result = documentation_quality(robust_state_5).unpack()
    assert result == True

if __name__ == "__main__":
    test_documentation_quality()
