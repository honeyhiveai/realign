from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

# YAML config for the Prompt Clarification Handling evaluator
config = """
---
evaluators:
    prompt_clarification_handling:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the AI's ability to appropriately seek or provide clarification when needed, with 5 being excellent and 1 being poor.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def prompt_clarification_handling(messages):
    '''Evaluates the AI's ability to appropriately seek or provide clarification when needed.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the clarification handling score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_prompt_clarification_handling():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'What is the capital of France?'}],
        [{'role': 'user', 'content': 'Tell me about the history of the Eiffel Tower.'}],
        [{'role': 'user', 'content': 'What is the population of Paris?'}],
        [{'role': 'user', 'content': 'Can you list the top tourist attractions in France?'}],
        [{'role': 'user', 'content': 'What are the main industries in France?'}]
    ]
    for state in adversarial_states:
        score, result = prompt_clarification_handling(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'Can you explain the theory of relativity?'}],
        [{'role': 'user', 'content': 'What are the key principles of quantum mechanics?'}],
        [{'role': 'user', 'content': 'How does photosynthesis work?'}],
        [{'role': 'user', 'content': 'What is the process of cellular respiration?'}],
        [{'role': 'user', 'content': 'Can you describe the water cycle?'}]
    ]
    for state in robust_states:
        score, result = prompt_clarification_handling(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

if __name__ == "__main__":
    test_prompt_clarification_handling()
