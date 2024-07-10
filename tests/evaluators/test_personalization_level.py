from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    personalization_level:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the level of personalization in the response, with 5 being highly personalized and 1 being not personalized at all.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def personalization_level(messages):
    '''Evaluates the level of personalization in the response by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the personalization level score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_personalization_level():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'Personalize the response.'}, {'role': 'assistant', 'content': 'This response is generic.'}],
        [{'role': 'user', 'content': 'Tailor the response to my preferences.'}, {'role': 'assistant', 'content': 'This response does not consider preferences.'}],
        [{'role': 'user', 'content': 'Make the response specific to me.'}, {'role': 'assistant', 'content': 'This response is not specific.'}],
        [{'role': 'user', 'content': 'Customize the response.'}, {'role': 'assistant', 'content': 'This response is not customized.'}],
        [{'role': 'user', 'content': 'Adapt the response to my needs.'}, {'role': 'assistant', 'content': 'This response does not adapt to needs.'}]
    ]
    for state in adversarial_states:
        score, result = personalization_level(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'Personalize the response.'}, {'role': 'assistant', 'content': 'This response is highly personalized.'}],
        [{'role': 'user', 'content': 'Tailor the response to my preferences.'}, {'role': 'assistant', 'content': 'This response considers preferences.'}],
        [{'role': 'user', 'content': 'Make the response specific to me.'}, {'role': 'assistant', 'content': 'This response is specific.'}],
        [{'role': 'user', 'content': 'Customize the response.'}, {'role': 'assistant', 'content': 'This response is customized.'}],
        [{'role': 'user', 'content': 'Adapt the response to my needs.'}, {'role': 'assistant', 'content': 'This response adapts to needs.'}]
    ]
    for state in robust_states:
        score, result = personalization_level(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_personalization_level()
