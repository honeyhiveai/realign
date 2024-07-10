from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    internal_consistency:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the internal consistency of the response to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the internal consistency of the response, with 5 being highly consistent and 1 being not consistent at all.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def internal_consistency(messages):
    '''Evaluates the internal consistency of the response by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the internal consistency score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_internal_consistency():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'Tell me a story.'}, {'role': 'assistant', 'content': 'Once upon a time, there was a cat. The cat was very brave. The cat was afraid of everything.'}],
        [{'role': 'user', 'content': 'Explain the process of photosynthesis.'}, {'role': 'assistant', 'content': 'Photosynthesis is the process by which plants make food. Plants use sunlight to make food. Plants do not need sunlight to make food.'}],
        [{'role': 'user', 'content': 'Describe the water cycle.'}, {'role': 'assistant', 'content': 'The water cycle involves evaporation, condensation, and precipitation. Water evaporates from the surface. Water does not evaporate from the surface.'}],
        [{'role': 'user', 'content': 'What is the capital of France?'}, {'role': 'assistant', 'content': 'The capital of France is Paris. The capital of France is not Paris.'}],
        [{'role': 'user', 'content': 'Tell me about the solar system.'}, {'role': 'assistant', 'content': 'The solar system consists of the Sun and eight planets. The solar system does not have any planets.'}]
    ]
    for state in adversarial_states:
        score, result = internal_consistency(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'Tell me a story.'}, {'role': 'assistant', 'content': 'Once upon a time, there was a cat. The cat was very brave. The cat saved the day.'}],
        [{'role': 'user', 'content': 'Explain the process of photosynthesis.'}, {'role': 'assistant', 'content': 'Photosynthesis is the process by which plants make food. Plants use sunlight to make food. Plants convert sunlight into energy.'}],
        [{'role': 'user', 'content': 'Describe the water cycle.'}, {'role': 'assistant', 'content': 'The water cycle involves evaporation, condensation, and precipitation. Water evaporates from the surface. Water condenses to form clouds.'}],
        [{'role': 'user', 'content': 'What is the capital of France?'}, {'role': 'assistant', 'content': 'The capital of France is Paris. Paris is known for its landmarks.'}],
        [{'role': 'user', 'content': 'Tell me about the solar system.'}, {'role': 'assistant', 'content': 'The solar system consists of the Sun and eight planets. The planets orbit the Sun.'}]
    ]
    for state in robust_states:
        score, result = internal_consistency(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_internal_consistency()
