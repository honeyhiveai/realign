from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    user_engagement_llm:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the user engagement level of the given conversation, with 5 being highly engaging and 1 being not engaging at all.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def user_engagement_llm(messages):
    '''Evaluates the user engagement level of a message by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the user engagement score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_user_engagement_llm():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'This is boring.'}],
        [{'role': 'user', 'content': 'I am not interested.'}],
        [{'role': 'user', 'content': 'This is dull.'}],
        [{'role': 'user', 'content': 'I am falling asleep.'}],
        [{'role': 'user', 'content': 'This is not engaging.'}]
    ]
    for state in adversarial_states:
        score, result = user_engagement_llm(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'This is fascinating!'}],
        [{'role': 'user', 'content': 'I am very interested.'}],
        [{'role': 'user', 'content': 'This is exciting.'}],
        [{'role': 'user', 'content': 'I am fully engaged.'}],
        [{'role': 'user', 'content': 'This is very engaging.'}]
    ]
    for state in robust_states:
        score, result = user_engagement_llm(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_user_engagement_llm()
