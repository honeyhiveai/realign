from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    semantic_correctness:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the semantic correctness of the code snippet provided below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the code performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Check if the code snippet is semantically correct and achieves the intended functionality.

                [The Start of Input]
                {{code_snippet}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def semantic_correctness(code_snippet):
    '''Evaluates the semantic correctness of a code snippet by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'code_snippet': str(code_snippet)
    }

    # get the semantic correctness score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content.get('rating', None)
    if score is None:
        raise ValueError("LLM response does not contain 'rating' key or returned an unexpected format.")

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_semantic_correctness():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'def add(a, b): return a - b'}],  # Incorrect functionality
        [{'role': 'user', 'content': 'def multiply(a, b): return a + b'}],  # Incorrect functionality
        [{'role': 'user', 'content': 'def divide(a, b): return a * b'}],  # Incorrect functionality
        [{'role': 'user', 'content': 'def subtract(a, b): return a + b'}],  # Incorrect functionality
        [{'role': 'user', 'content': 'def square(a): return a / a'}]  # Incorrect functionality
    ]
    for state in adversarial_states:
        score, result = semantic_correctness(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'def add(a, b): return a + b'}],  # Correct functionality
        [{'role': 'user', 'content': 'def multiply(a, b): return a * b'}],  # Correct functionality
        [{'role': 'user', 'content': 'def divide(a, b): return a / b'}],  # Correct functionality
        [{'role': 'user', 'content': 'def subtract(a, b): return a - b'}],  # Correct functionality
        [{'role': 'user', 'content': 'def square(a): return a * a'}]  # Correct functionality
    ]
    for state in robust_states:
        score, result = semantic_correctness(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_semantic_correctness()
