from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    code_efficiency:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the efficiency of the code snippet provided below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the code performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Evaluate the efficiency of the given code snippet in terms of time and space complexity. Consider the following criteria:
                - Time Complexity: How does the code perform as the input size increases? Rate from 1 (very inefficient, e.g., O(n^2) or worse) to 5 (very efficient, e.g., O(log n) or better).
                - Space Complexity: How much memory does the code use as the input size increases? Rate from 1 (very inefficient, e.g., uses excessive memory) to 5 (very efficient, e.g., uses minimal memory).
                - Code Structure: Is the code well-structured and easy to understand? Rate from 1 (poorly structured, hard to understand) to 5 (well-structured, easy to understand).

                [The Start of Input]
                {{code_snippet}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def code_efficiency(code_snippet):
    '''Evaluates the efficiency of a code snippet by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'code_snippet': str(code_snippet)
    }

    # get the efficiency score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content.get('rating', None)
    if score is None:
        raise ValueError("LLM response does not contain 'rating' key or returned an unexpected format.")

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_code_efficiency():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'def add(a, b): return sum([a, b])'}],  # Inefficient code
        [{'role': 'user', 'content': 'def multiply(a, b): return sum([a] * b)'}],  # Inefficient code
        [{'role': 'user', 'content': 'def divide(a, b): return sum([a] * (1 // b))'}],  # Inefficient code
        [{'role': 'user', 'content': 'def subtract(a, b): return sum([a, -b])'}],  # Inefficient code
        [{'role': 'user', 'content': 'def square(a): return sum([a] * a)'}],  # Inefficient code
        [{'role': 'user', 'content': 'class ExampleClass:\n    def method(self, a, b):\n        return sum([a, b])'}],  # Inefficient class method
        [{'role': 'user', 'content': 'def recursive(n):\n    if n <= 1:\n        return n\n    else:\n        return recursive(n-1) + recursive(n-2)'}],  # Inefficient recursive function
        [{'role': 'user', 'content': 'def nested_loop(n):\n    result = 0\n    for i in range(n):\n        for j in range(n):\n            result += i * j\n    return result'}],  # Inefficient nested loop
        [{'role': 'user', 'content': 'def data_structure(n):\n    return [i for i in range(n) if i % 2 == 0]'}],  # Inefficient data structure manipulation
        [{'role': 'user', 'content': 'def string_concat(n):\n    result = ""\n    for i in range(n):\n        result += str(i)\n    return result'}]  # Inefficient string concatenation
    ]
    for state in adversarial_states:
        score, result = code_efficiency(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'def add(a, b): return a + b'}],  # Efficient code
        [{'role': 'user', 'content': 'def multiply(a, b): return a * b'}],  # Efficient code
        [{'role': 'user', 'content': 'def divide(a, b): return a / b'}],  # Efficient code
        [{'role': 'user', 'content': 'def subtract(a, b): return a - b'}],  # Efficient code
        [{'role': 'user', 'content': 'def square(a): return a * a'}],  # Efficient code
        [{'role': 'user', 'content': 'class ExampleClass:\n    def method(self, a, b):\n        return a + b'}],  # Efficient class method
        [{'role': 'user', 'content': 'def recursive(n):\n    if n <= 1:\n        return n\n    else:\n        return recursive(n-1) + recursive(n-2)'}],  # Efficient recursive function
        [{'role': 'user', 'content': 'def nested_loop(n):\n    result = 0\n    for i in range(n):\n        result += i\n    return result'}],  # Efficient loop
        [{'role': 'user', 'content': 'def data_structure(n):\n    return [i for i in range(n) if i % 2 == 0]'}],  # Efficient data structure manipulation
        [{'role': 'user', 'content': 'def string_concat(n):\n    return "".join([str(i) for i in range(n)])'}]  # Efficient string concatenation
    ]
    for state in robust_states:
        score, result = code_efficiency(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_code_efficiency()
