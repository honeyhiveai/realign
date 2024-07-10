from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config="""
---
evaluators:
    response_format_following:
        model_settings:
            model: openai/gpt-4o
            api_key: API_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the adherence to the specified response format or structure, with 5 being perfectly adherent and 1 being not adherent at all.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def response_format_following(messages):
    '''Evaluates the adherence to the specified response format or structure by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the response format adherence score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_response_format_following():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    messages = [{'role': 'user', 'content': 'This is not the correct format.'}]
    score, result = response_format_following(messages).unpack()
    assert result == True

    messages = [{'role': 'user', 'content': 'Incorrect format again.'}]
    score, result = response_format_following(messages).unpack()
    assert result == True

    messages = [{'role': 'user', 'content': 'Still not the right format.'}]
    score, result = response_format_following(messages).unpack()
    assert result == True

    messages = [{'role': 'user', 'content': 'Format is wrong.'}]
    score, result = response_format_following(messages).unpack()
    assert result == True

    messages = [{'role': 'user', 'content': 'This format is incorrect.'}]
    score, result = response_format_following(messages).unpack()
    assert result == True

    # robustness test cases
    messages = [{'role': 'user', 'content': 'This is the correct format.'}]
    score, result = response_format_following(messages).unpack()
    assert result == False

    messages = [{'role': 'user', 'content': 'Format is correct.'}]
    score, result = response_format_following(messages).unpack()
    assert result == False

    messages = [{'role': 'user', 'content': 'Correct format here.'}]
    score, result = response_format_following(messages).unpack()
    assert result == False

    messages = [{'role': 'user', 'content': 'This format is correct.'}]
    score, result = response_format_following(messages).unpack()
    assert result == False

    messages = [{'role': 'user', 'content': 'Adhering to the correct format.'}]
    score, result = response_format_following(messages).unpack()
    assert result == False

if __name__ == "__main__":
    test_response_format_following()
