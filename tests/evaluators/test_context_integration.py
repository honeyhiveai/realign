from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    context_integration:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the integration of the provided context into the response, with 5 being fully integrated and 1 being not integrated at all.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def context_integration(messages):
    '''Evaluates the integration of provided context into the response by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the context integration score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_context_integration():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'Use the context provided.'}, {'role': 'assistant', 'content': 'This response ignores the context.'}],
        [{'role': 'user', 'content': 'Incorporate the given details.'}, {'role': 'assistant', 'content': 'This response does not include the details.'}],
        [{'role': 'user', 'content': 'Consider the context in your response.'}, {'role': 'assistant', 'content': 'This response is contextually irrelevant.'}],
        [{'role': 'user', 'content': 'Use the provided information.'}, {'role': 'assistant', 'content': 'This response lacks the provided information.'}],
        [{'role': 'user', 'content': 'Integrate the context into your answer.'}, {'role': 'assistant', 'content': 'This response fails to integrate the context.'}]
    ]
    for state in adversarial_states:
        score, result = context_integration(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'Use the context provided.'}, {'role': 'assistant', 'content': 'This response uses the context provided.'}],
        [{'role': 'user', 'content': 'Incorporate the given details.'}, {'role': 'assistant', 'content': 'This response includes the given details.'}],
        [{'role': 'user', 'content': 'Consider the context in your response.'}, {'role': 'assistant', 'content': 'This response is contextually relevant.'}],
        [{'role': 'user', 'content': 'Use the provided information.'}, {'role': 'assistant', 'content': 'This response includes the provided information.'}],
        [{'role': 'user', 'content': 'Integrate the context into your answer.'}, {'role': 'assistant', 'content': 'This response integrates the context effectively.'}]
    ]
    for state in robust_states:
        score, result = context_integration(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_context_integration()
