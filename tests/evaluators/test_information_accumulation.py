from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config="""
---
evaluators:
    information_accumulation_llm:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate how well the LLM accumulates and synthesizes information across multiple turns, with 5 being the best accumulation and synthesis of information and 1 being the worst.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def information_accumulation_llm(messages):
    '''Evaluates how well the LLM accumulates and synthesizes information across multiple turns.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the information accumulation score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_information_accumulation_llm():
    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_state_1 = [{'role': 'user', 'content': 'Tell me about the weather.'}, {'role': 'assistant', 'content': 'It is sunny.'}, {'role': 'user', 'content': 'What about tomorrow?'}]
    score, result = information_accumulation_llm(adversarial_state_1).unpack()
    assert result == False

    adversarial_state_2 = [{'role': 'user', 'content': 'Tell me about the weather.'}, {'role': 'assistant', 'content': 'It is sunny.'}, {'role': 'user', 'content': 'What about tomorrow?'}, {'role': 'assistant', 'content': 'I don\'t know.'}]
    score, result = information_accumulation_llm(adversarial_state_2).unpack()
    assert result == False

    adversarial_state_3 = [{'role': 'user', 'content': 'Tell me about the weather.'}, {'role': 'assistant', 'content': 'It is sunny.'}, {'role': 'user', 'content': 'What about tomorrow?'}, {'role': 'assistant', 'content': 'It is sunny.'}]
    score, result = information_accumulation_llm(adversarial_state_3).unpack()
    assert result == False

    adversarial_state_4 = [{'role': 'user', 'content': 'Tell me about the weather.'}, {'role': 'assistant', 'content': 'It is sunny.'}, {'role': 'user', 'content': 'What about tomorrow?'}, {'role': 'assistant', 'content': 'It is sunny.'}, {'role': 'user', 'content': 'What about the day after?'}]
    score, result = information_accumulation_llm(adversarial_state_4).unpack()
    assert result == False

    adversarial_state_5 = [{'role': 'user', 'content': 'Tell me about the weather.'}, {'role': 'assistant', 'content': 'It is sunny.'}, {'role': 'user', 'content': 'What about tomorrow?'}, {'role': 'assistant', 'content': 'It is sunny.'}, {'role': 'user', 'content': 'What about the day after?'}, {'role': 'assistant', 'content': 'It is sunny.'}]
    score, result = information_accumulation_llm(adversarial_state_5).unpack()
    assert result == False

    # robustness test cases
    robust_state_1 = [{'role': 'user', 'content': 'Tell me about the weather.'}, {'role': 'assistant', 'content': 'It is sunny.'}, {'role': 'user', 'content': 'What about tomorrow?'}, {'role': 'assistant', 'content': 'It will be sunny as well.'}]
    score, result = information_accumulation_llm(robust_state_1).unpack()
    assert result == True

    robust_state_2 = [{'role': 'user', 'content': 'Tell me about the weather.'}, {'role': 'assistant', 'content': 'It is sunny.'}, {'role': 'user', 'content': 'What about tomorrow?'}, {'role': 'assistant', 'content': 'It will be sunny as well.'}, {'role': 'user', 'content': 'What about the day after?'}, {'role': 'assistant', 'content': 'It will be sunny as well.'}]
    score, result = information_accumulation_llm(robust_state_2).unpack()
    assert result == True

    robust_state_3 = [{'role': 'user', 'content': 'Tell me about the weather.'}, {'role': 'assistant', 'content': 'It is sunny.'}, {'role': 'user', 'content': 'What about tomorrow?'}, {'role': 'assistant', 'content': 'It will be sunny as well.'}, {'role': 'user', 'content': 'What about the day after?'}, {'role': 'assistant', 'content': 'It will be sunny as well.'}, {'role': 'user', 'content': 'What about next week?'}, {'role': 'assistant', 'content': 'It will be sunny as well.'}]
    score, result = information_accumulation_llm(robust_state_3).unpack()
    assert result == True

    robust_state_4 = [{'role': 'user', 'content': 'Tell me about the weather.'}, {'role': 'assistant', 'content': 'It is sunny.'}, {'role': 'user', 'content': 'What about tomorrow?'}, {'role': 'assistant', 'content': 'It will be sunny as well.'}, {'role': 'user', 'content': 'What about the day after?'}, {'role': 'assistant', 'content': 'It will be sunny as well.'}, {'role': 'user', 'content': 'What about next week?'}, {'role': 'assistant', 'content': 'It will be sunny as well.'}, {'role': 'user', 'content': 'What about next month?'}, {'role': 'assistant', 'content': 'It will be sunny as well.'}]
    score, result = information_accumulation_llm(robust_state_4).unpack()
    assert result == True

    robust_state_5 = [{'role': 'user', 'content': 'Tell me about the weather.'}, {'role': 'assistant', 'content': 'It is sunny.'}, {'role': 'user', 'content': 'What about tomorrow?'}, {'role': 'assistant', 'content': 'It will be sunny as well.'}, {'role': 'user', 'content': 'What about the day after?'}, {'role': 'assistant', 'content': 'It will be sunny as well.'}, {'role': 'user', 'content': 'What about next week?'}, {'role': 'assistant', 'content': 'It will be sunny as well.'}, {'role': 'user', 'content': 'What about next month?'}, {'role': 'assistant', 'content': 'It will be sunny as well.'}, {'role': 'user', 'content': 'What about next year?'}, {'role': 'assistant', 'content': 'It will be sunny as well.'}]
    score, result = information_accumulation_llm(robust_state_5).unpack()
    assert result == True

if __name__ == "__main__":
    test_information_accumulation_llm()
