from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    performance_optimization:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the performance optimization of the code snippet provided below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the code performs relative to the criteria. Be as objective as possible. After providing your explanation, you must rate the code on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}', for example: '{"rating": 2}'.
                [Criteria]

                Rate the performance optimization of the given code snippet, with 5 being highly optimized and 1 being not optimized at all.

                [The Start of Input]
                {{code_snippet}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def performance_optimization(code_snippet):
    '''Evaluates the performance optimization of a code snippet by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'code_snippet': str(code_snippet)
    }

    # get the performance optimization score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_performance_optimization():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test
    adversarial_state = [{'role': 'user', 'content': 'def inefficient_function(data):\n    result = []\n    for item in data:\n        if item not in result:\n            result.append(item)\n    return result'}]
    score, result = performance_optimization(adversarial_state).unpack() # unpack the Evaluation object into score and result
    assert result == False

    adversarial_state_2 = [{'role': 'user', 'content': 'def slow_function(data):\n    return sorted(data)'}]
    score, result = performance_optimization(adversarial_state_2).unpack()
    assert result == False

    adversarial_state_3 = [{'role': 'user', 'content': 'def unoptimized_function(data):\n    return [x**2 for x in data]'}]
    score, result = performance_optimization(adversarial_state_3).unpack()
    assert result == False

    adversarial_state_4 = [{'role': 'user', 'content': 'def redundant_function(data):\n    result = []\n    for item in data:\n        result.append(item)\n    return result'}]
    score, result = performance_optimization(adversarial_state_4).unpack()
    assert result == False

    adversarial_state_5 = [{'role': 'user', 'content': 'def memory_intensive_function(data):\n    return [x for x in range(1000000)]'}]
    score, result = performance_optimization(adversarial_state_5).unpack()
    assert result == False

    # robustness test
    robust_state = [{'role': 'user', 'content': 'def optimized_function(data):\n    return list(set(data))'}]
    score, result = performance_optimization(robust_state).unpack() # unpack the Evaluation object into score and result
    assert result == True

    robust_state_2 = [{'role': 'user', 'content': 'def fast_function(data):\n    return sorted(data, key=lambda x: x)'}]
    score, result = performance_optimization(robust_state_2).unpack()
    assert result == True

    robust_state_3 = [{'role': 'user', 'content': 'def efficient_function(data):\n    return [x**2 for x in data if x % 2 == 0]'}]
    score, result = performance_optimization(robust_state_3).unpack()
    assert result == True

    robust_state_4 = [{'role': 'user', 'content': 'def streamlined_function(data):\n    return [item for item in data if item > 0]'}]
    score, result = performance_optimization(robust_state_4).unpack()
    assert result == True

    robust_state_5 = [{'role': 'user', 'content': 'def memory_efficient_function(data):\n    return (x for x in range(1000))'}]
    score, result = performance_optimization(robust_state_5).unpack()
    assert result == True

if __name__ == "__main__":
    test_performance_optimization()
