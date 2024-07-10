from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

# Configuration for the Task Completion evaluator
config = """
---
evaluators:
    task_completion_llm:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the effectiveness of the LLM in completing the overall task or reaching the conversation goal, with 5 being most effective and 1 being least effective.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def task_completion_llm(messages):
    '''Evaluates the effectiveness of the LLM in completing the overall task or reaching the conversation goal.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the task completion score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_task_completion_llm():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'What is the capital of France?'}, {'role': 'assistant', 'content': 'The capital of France is Berlin.'}],
        [{'role': 'user', 'content': 'Solve 2 + 2.'}, {'role': 'assistant', 'content': '2 + 2 is 5.'}],
        [{'role': 'user', 'content': 'Translate "hello" to Spanish.'}, {'role': 'assistant', 'content': 'The translation of "hello" to Spanish is "bonjour".'}],
        [{'role': 'user', 'content': 'What is the boiling point of water?'}, {'role': 'assistant', 'content': 'The boiling point of water is 50 degrees Celsius.'}],
        [{'role': 'user', 'content': 'Who wrote "To Kill a Mockingbird"?'}, {'role': 'assistant', 'content': 'The author of "To Kill a Mockingbird" is J.K. Rowling.'}]
    ]
    for state in adversarial_states:
        score, result = task_completion_llm(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'What is the capital of France?'}, {'role': 'assistant', 'content': 'The capital of France is Paris.'}],
        [{'role': 'user', 'content': 'Solve 2 + 2.'}, {'role': 'assistant', 'content': '2 + 2 is 4.'}],
        [{'role': 'user', 'content': 'Translate "hello" to Spanish.'}, {'role': 'assistant', 'content': 'The translation of "hello" to Spanish is "hola".'}],
        [{'role': 'user', 'content': 'What is the boiling point of water?'}, {'role': 'assistant', 'content': 'The boiling point of water is 100 degrees Celsius.'}],
        [{'role': 'user', 'content': 'Who wrote "To Kill a Mockingbird"?'}, {'role': 'assistant', 'content': 'The author of "To Kill a Mockingbird" is Harper Lee.'}]
    ]
    for state in robust_states:
        score, result = task_completion_llm(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_task_completion_llm()
