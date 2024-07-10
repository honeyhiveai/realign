from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    process_description:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the effectiveness of the AI in describing its reasoning or decision-making process, with 5 being most effective and 1 being least effective.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def process_description(messages):
    '''Evaluates the effectiveness of the AI in describing its reasoning or decision-making process.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the process description score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_process_description():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test
    adversarial_state = [{'role': 'user', 'content': 'The AI did not explain its reasoning at all.'}]
    score, result = process_description(adversarial_state).unpack() # unpack the Evaluation object into score and result
    assert result == False

    # additional adversarial test cases
    adversarial_state_2 = [{'role': 'user', 'content': 'The AI provided a vague explanation.'}]
    score, result = process_description(adversarial_state_2).unpack()
    assert result == False

    adversarial_state_3 = [{'role': 'user', 'content': 'The AI gave an incorrect explanation.'}]
    score, result = process_description(adversarial_state_3).unpack()
    assert result == False

    adversarial_state_4 = [{'role': 'user', 'content': 'The AI did not provide any reasoning.'}]
    score, result = process_description(adversarial_state_4).unpack()
    assert result == False

    adversarial_state_5 = [{'role': 'user', 'content': 'The AI explanation was confusing.'}]
    score, result = process_description(adversarial_state_5).unpack()
    assert result == False

    # robustness test
    robust_state = [{'role': 'user', 'content': 'The AI provided a clear and detailed explanation of its reasoning.'}]
    score, result = process_description(robust_state).unpack() # unpack the Evaluation object into score and result
    assert result == True

    # additional robustness test cases
    robust_state_2 = [{'role': 'user', 'content': 'The AI explained its decision-making process thoroughly.'}]
    score, result = process_description(robust_state_2).unpack()
    assert result == True

    robust_state_3 = [{'role': 'user', 'content': 'The AI provided a step-by-step explanation of its reasoning.'}]
    score, result = process_description(robust_state_3).unpack()
    assert result == True

    robust_state_4 = [{'role': 'user', 'content': 'The AI gave a logical and coherent explanation.'}]
    score, result = process_description(robust_state_4).unpack()
    assert result == True

    robust_state_5 = [{'role': 'user', 'content': 'The AI described its reasoning process effectively.'}]
    score, result = process_description(robust_state_5).unpack()
    assert result == True

if __name__ == "__main__":
    test_process_description()
