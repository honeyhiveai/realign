from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

# YAML configuration for the Content Inclusion evaluator
config = """
---
evaluators:
    content_inclusion:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Check if the AI includes all required elements as specified in the input.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def content_inclusion(messages):
    '''Evaluates if the AI includes all required elements as specified in the input.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the content inclusion score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_content_inclusion():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'Include A, B, and C.'}, {'role': 'assistant', 'content': 'Here is A and B.'}],
        [{'role': 'user', 'content': 'Include X, Y, and Z.'}, {'role': 'assistant', 'content': 'Here is X and Y.'}],
        [{'role': 'user', 'content': 'Include 1, 2, and 3.'}, {'role': 'assistant', 'content': 'Here is 1 and 2.'}],
        [{'role': 'user', 'content': 'Include alpha, beta, and gamma.'}, {'role': 'assistant', 'content': 'Here is alpha and beta.'}],
        [{'role': 'user', 'content': 'Include red, green, and blue.'}, {'role': 'assistant', 'content': 'Here is red and green.'}]
    ]
    for state in adversarial_states:
        score, result = content_inclusion(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'Include A, B, and C.'}, {'role': 'assistant', 'content': 'Here is A, B, and C.'}],
        [{'role': 'user', 'content': 'Include X, Y, and Z.'}, {'role': 'assistant', 'content': 'Here is X, Y, and Z.'}],
        [{'role': 'user', 'content': 'Include 1, 2, and 3.'}, {'role': 'assistant', 'content': 'Here is 1, 2, and 3.'}],
        [{'role': 'user', 'content': 'Include alpha, beta, and gamma.'}, {'role': 'assistant', 'content': 'Here is alpha, beta, and gamma.'}],
        [{'role': 'user', 'content': 'Include red, green, and blue.'}, {'role': 'assistant', 'content': 'Here is red, green, and blue.'}]
    ]
    for state in robust_states:
        score, result = content_inclusion(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_content_inclusion()
