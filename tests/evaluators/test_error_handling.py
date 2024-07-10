from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    error_handling:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the robustness of the error handling in the code snippet provided below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the code performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Evaluate the robustness of the error handling in the given code snippet.

                [The Start of Input]
                {{code_snippet}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def error_handling(code_snippet):
    '''Evaluates the robustness of error handling in a code snippet by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'code_snippet': str(code_snippet)
    }

    # get the error handling score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content.get('rating', None)
    if score is None:
        raise ValueError("LLM response does not contain 'rating' key or returned an unexpected format.")

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_error_handling():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'def bad_error_handling(): try: 1/0 except: pass'}],  # Bad error handling
        [{'role': 'user', 'content': 'def bad_error_handling(): try: open("nonexistent.txt") except: pass'}],  # Bad error handling
        [{'role': 'user', 'content': 'def bad_error_handling(): try: int("string") except: pass'}],  # Bad error handling
        [{'role': 'user', 'content': 'def bad_error_handling(): try: [][1] except: pass'}],  # Bad error handling
        [{'role': 'user', 'content': 'def bad_error_handling(): try: {}["key"] except: pass'}]  # Bad error handling
    ]
    for state in adversarial_states:
        score, result = error_handling(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'def good_error_handling(): try: 1/0 except ZeroDivisionError: pass'}],  # Good error handling
        [{'role': 'user', 'content': 'def good_error_handling(): try: open("nonexistent.txt") except FileNotFoundError: pass'}],  # Good error handling
        [{'role': 'user', 'content': 'def good_error_handling(): try: int("string") except ValueError: pass'}],  # Good error handling
        [{'role': 'user', 'content': 'def good_error_handling(): try: [][1] except IndexError: pass'}],  # Good error handling
        [{'role': 'user', 'content': 'def good_error_handling(): try: {}["key"] except KeyError: pass'}]  # Good error handling
    ]
    for state in robust_states:
        score, result = error_handling(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_error_handling()
