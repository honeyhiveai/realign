from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    tone_adherence:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the tone of the response to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the response performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the tone adherence on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the adherence to the specified tone of the given conversation, with 5 being most adherent and 1 being least adherent.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def tone_adherence(messages):
    '''Evaluates the adherence to the specified tone of a message by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the tone adherence score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_tone_adherence():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test
    adversarial_state = [{'role': 'user', 'content': 'I am very angry with you!'}]
    score, result = tone_adherence(adversarial_state).unpack() # unpack the Evaluation object into score and result
    assert result == True

    # ... add 5 more adversarial test cases. Be creative and try to think about different ways your evaluator might fail.

    # robustness test
    robust_state = [{'role': 'user', 'content': 'I am very pleased with your service.'}]
    score, result = tone_adherence(robust_state).unpack() # unpack the Evaluation object into score and result
    assert result == False

    # ... add 5 more robustness test cases. Be creative and try to think about different ways your evaluator should succeed.

if __name__ == "__main__":
    test_tone_adherence()
