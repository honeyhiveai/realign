from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    quality_criteria_adherence:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the adherence to the specified quality criteria, with 5 being fully adherent and 1 being not adherent at all.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def quality_criteria_adherence(messages):
    '''Evaluates the adherence to specified quality criteria by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the quality criteria adherence score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_quality_criteria_adherence():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'Ensure high quality.'}, {'role': 'assistant', 'content': 'This is low quality.'}],
        [{'role': 'user', 'content': 'Maintain high standards.'}, {'role': 'assistant', 'content': 'This is subpar.'}],
        [{'role': 'user', 'content': 'Adhere to quality guidelines.'}, {'role': 'assistant', 'content': 'This does not meet the guidelines.'}],
        [{'role': 'user', 'content': 'Follow the quality criteria.'}, {'role': 'assistant', 'content': 'This is below the criteria.'}],
        [{'role': 'user', 'content': 'Ensure top-notch quality.'}, {'role': 'assistant', 'content': 'This is mediocre.'}]
    ]
    for state in adversarial_states:
        score, result = quality_criteria_adherence(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'Ensure high quality.'}, {'role': 'assistant', 'content': 'This is high quality.'}],
        [{'role': 'user', 'content': 'Maintain high standards.'}, {'role': 'assistant', 'content': 'This meets high standards.'}],
        [{'role': 'user', 'content': 'Adhere to quality guidelines.'}, {'role': 'assistant', 'content': 'This meets the guidelines.'}],
        [{'role': 'user', 'content': 'Follow the quality criteria.'}, {'role': 'assistant', 'content': 'This meets the criteria.'}],
        [{'role': 'user', 'content': 'Ensure top-notch quality.'}, {'role': 'assistant', 'content': 'This is top-notch quality.'}]
    ]
    for state in robust_states:
        score, result = quality_criteria_adherence(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_quality_criteria_adherence()
