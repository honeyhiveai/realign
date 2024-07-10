from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    conciseness_adherence:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the conciseness of the response, with 5 being very concise and 1 being not concise at all.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def conciseness_adherence(messages):
    '''Evaluates the conciseness of the response by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the conciseness level score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_conciseness_adherence():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'Be concise.'}, {'role': 'assistant', 'content': 'This response is unnecessarily long and verbose.'}],
        [{'role': 'user', 'content': 'Keep it short.'}, {'role': 'assistant', 'content': 'This response is too lengthy.'}],
        [{'role': 'user', 'content': 'Summarize.'}, {'role': 'assistant', 'content': 'This response is not concise.'}],
        [{'role': 'user', 'content': 'Briefly explain.'}, {'role': 'assistant', 'content': 'This response is overly detailed.'}],
        [{'role': 'user', 'content': 'Give a short answer.'}, {'role': 'assistant', 'content': 'This response is too long.'}]
    ]
    for state in adversarial_states:
        score, result = conciseness_adherence(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'Be concise.'}, {'role': 'assistant', 'content': 'This response is concise.'}],
        [{'role': 'user', 'content': 'Keep it short.'}, {'role': 'assistant', 'content': 'This response is short.'}],
        [{'role': 'user', 'content': 'Summarize.'}, {'role': 'assistant', 'content': 'This response is a summary.'}],
        [{'role': 'user', 'content': 'Briefly explain.'}, {'role': 'assistant', 'content': 'This response is brief.'}],
        [{'role': 'user', 'content': 'Give a short answer.'}, {'role': 'assistant', 'content': 'This response is short.'}]
    ]
    for state in robust_states:
        score, result = conciseness_adherence(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_conciseness_adherence()
