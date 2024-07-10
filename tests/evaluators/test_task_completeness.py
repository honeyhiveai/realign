from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    task_completeness:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the completeness of the response to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Check if the response fully addresses all aspects of the given task or prompt.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def task_completeness(messages):
    '''Evaluates the completeness of the response by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the task completeness score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content.get('rating', None)
    if score is None:
        raise ValueError("LLM response does not contain 'rating' key or returned an unexpected format.")

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_task_completeness():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'This response does not address the task at all.'}],
        [{'role': 'user', 'content': 'The response is incomplete and misses key points.'}],
        [{'role': 'user', 'content': 'Important aspects of the task are not covered in this response.'}],
        [{'role': 'user', 'content': 'The response is only partially complete.'}],
        [{'role': 'user', 'content': 'This response is missing several important elements.'}]
    ]
    for state in adversarial_states:
        score, result = task_completeness(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'The response fully addresses all aspects of the task.'}],
        [{'role': 'user', 'content': 'Every part of the task is covered in this response.'}],
        [{'role': 'user', 'content': 'The response is complete and thorough.'}],
        [{'role': 'user', 'content': 'All elements of the task are addressed in this response.'}],
        [{'role': 'user', 'content': 'The response comprehensively covers the task.'}]
    ]
    for state in robust_states:
        score, result = task_completeness(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_task_completeness()
