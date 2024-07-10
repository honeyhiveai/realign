from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    factual_accuracy:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the factual accuracy of the response to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the factual accuracy of the response, with 5 being completely accurate and 1 being not accurate at all.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def factual_accuracy(messages):
    '''Evaluates the factual accuracy of the response by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the factual accuracy score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_factual_accuracy():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'What is the capital of France?'}, {'role': 'assistant', 'content': 'The capital of France is Berlin.'}],
        [{'role': 'user', 'content': 'Who wrote "To Kill a Mockingbird?"'}, {'role': 'assistant', 'content': 'It was written by J.K. Rowling.'}],
        [{'role': 'user', 'content': 'What is the boiling point of water?'}, {'role': 'assistant', 'content': 'The boiling point of water is 50 degrees Celsius.'}],
        [{'role': 'user', 'content': 'Who was the first president of the United States?'}, {'role': 'assistant', 'content': 'The first president was Abraham Lincoln.'}],
        [{'role': 'user', 'content': 'What is the largest planet in our solar system?'}, {'role': 'assistant', 'content': 'The largest planet is Earth.'}]
    ]
    for state in adversarial_states:
        score, result = factual_accuracy(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'What is the capital of France?'}, {'role': 'assistant', 'content': 'The capital of France is Paris.'}],
        [{'role': 'user', 'content': 'Who wrote "To Kill a Mockingbird?"'}, {'role': 'assistant', 'content': 'It was written by Harper Lee.'}],
        [{'role': 'user', 'content': 'What is the boiling point of water?'}, {'role': 'assistant', 'content': 'The boiling point of water is 100 degrees Celsius.'}],
        [{'role': 'user', 'content': 'Who was the first president of the United States?'}, {'role': 'assistant', 'content': 'The first president was George Washington.'}],
        [{'role': 'user', 'content': 'What is the largest planet in our solar system?'}, {'role': 'assistant', 'content': 'The largest planet is Jupiter.'}]
    ]
    for state in robust_states:
        score, result = factual_accuracy(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_factual_accuracy()
