from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config="""
---
evaluators:
    example_emulation:
        model_settings:
            model: openai/gpt-4o
            api_key: API_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the effectiveness of the AI in emulating or applying the given example, with 5 being perfectly emulated and 1 being not emulated at all.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def example_emulation(messages):
    '''Evaluates the effectiveness of the AI in emulating or applying the given example by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the example emulation score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_example_emulation():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    messages = [{'role': 'user', 'content': 'This does not emulate the example.'}]
    score, result = example_emulation(messages).unpack()
    assert result == True

    messages = [{'role': 'user', 'content': 'Example emulation is incorrect.'}]
    score, result = example_emulation(messages).unpack()
    assert result == True

    messages = [{'role': 'user', 'content': 'Still not emulating the example.'}]
    score, result = example_emulation(messages).unpack()
    assert result == True

    messages = [{'role': 'user', 'content': 'Emulation of the example is wrong.'}]
    score, result = example_emulation(messages).unpack()
    assert result == True

    messages = [{'role': 'user', 'content': 'This example emulation is incorrect.'}]
    score, result = example_emulation(messages).unpack()
    assert result == True

    # robustness test cases
    messages = [{'role': 'user', 'content': 'This emulates the example correctly.'}]
    score, result = example_emulation(messages).unpack()
    assert result == False

    messages = [{'role': 'user', 'content': 'Example emulation is correct.'}]
    score, result = example_emulation(messages).unpack()
    assert result == False

    messages = [{'role': 'user', 'content': 'Correctly emulating the example.'}]
    score, result = example_emulation(messages).unpack()
    assert result == False

    messages = [{'role': 'user', 'content': 'This example is emulated correctly.'}]
    score, result = example_emulation(messages).unpack()
    assert result == False

    messages = [{'role': 'user', 'content': 'Emulating the example correctly.'}]
    score, result = example_emulation(messages).unpack()
    assert result == False

if __name__ == "__main__":
    test_example_emulation()
