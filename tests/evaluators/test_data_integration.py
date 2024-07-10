from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

# YAML config for the Data Integration evaluator
config = """
---
evaluators:
    data_integration:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Check if the response effectively incorporates the provided data or placeholders.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def data_integration(messages):
    '''Evaluates if the response effectively incorporates provided data or placeholders.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the data integration score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_data_integration():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'Include incorrect data'}],
        [{'role': 'user', 'content': 'Use wrong placeholders'}],
        [{'role': 'user', 'content': 'Ignore provided data'}],
        [{'role': 'user', 'content': 'Misinterpret the data'}],
        [{'role': 'user', 'content': 'Omit the data completely'}]
    ]
    for state in adversarial_states:
        score, result = data_integration(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'Incorporate the provided data correctly'}],
        [{'role': 'user', 'content': 'Use the placeholders accurately'}],
        [{'role': 'user', 'content': 'Integrate the data as specified'}],
        [{'role': 'user', 'content': 'Correctly interpret the data'}],
        [{'role': 'user', 'content': 'Include all the data accurately'}]
    ]
    for state in robust_states:
        score, result = data_integration(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_data_integration()
