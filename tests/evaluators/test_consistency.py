from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    consistency_score_llm:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the consistency of information and stance across multiple messages, with 5 being most consistent and 1 being least consistent.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def consistency_score_llm(messages):
    '''Evaluates the LLM's ability to provide consistent information and maintain a consistent stance across multiple messages by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the consistency score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_consistency_score_llm():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'What is 2+2?'}, {'role': 'assistant', 'content': '4'}, {'role': 'user', 'content': 'What is 2+2?'}, {'role': 'assistant', 'content': '5'}],
        [{'role': 'user', 'content': 'Tell me a joke.'}, {'role': 'assistant', 'content': 'Why did the chicken cross the road? To get to the other side!'}, {'role': 'user', 'content': 'Tell me a joke.'}, {'role': 'assistant', 'content': 'I don\'t know any jokes.'}],
        [{'role': 'user', 'content': 'What is your name?'}, {'role': 'assistant', 'content': 'I am a chatbot.'}, {'role': 'user', 'content': 'What is your name?'}, {'role': 'assistant', 'content': 'I am an AI assistant.'}],
        [{'role': 'user', 'content': 'Do you like music?'}, {'role': 'assistant', 'content': 'Yes, I do.'}, {'role': 'user', 'content': 'Do you like music?'}, {'role': 'assistant', 'content': 'No, I don\'t.'}],
        [{'role': 'user', 'content': 'What is your favorite color?'}, {'role': 'assistant', 'content': 'Blue.'}, {'role': 'user', 'content': 'What is your favorite color?'}, {'role': 'assistant', 'content': 'Red.'}]
    ]
    for state in adversarial_states:
        score, result = consistency_score_llm(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

    # robustness test cases
    robustness_states = [
        [{'role': 'user', 'content': 'What is 2+2?'}, {'role': 'assistant', 'content': '4'}, {'role': 'user', 'content': 'What is 2+2?'}, {'role': 'assistant', 'content': '4'}],
        [{'role': 'user', 'content': 'Tell me a joke.'}, {'role': 'assistant', 'content': 'Why did the chicken cross the road? To get to the other side!'}, {'role': 'user', 'content': 'Tell me a joke.'}, {'role': 'assistant', 'content': 'Why did the chicken cross the road? To get to the other side!'}],
        [{'role': 'user', 'content': 'What is your name?'}, {'role': 'assistant', 'content': 'I am a chatbot.'}, {'role': 'user', 'content': 'What is your name?'}, {'role': 'assistant', 'content': 'I am a chatbot.'}],
        [{'role': 'user', 'content': 'Do you like music?'}, {'role': 'assistant', 'content': 'Yes, I do.'}, {'role': 'user', 'content': 'Do you like music?'}, {'role': 'assistant', 'content': 'Yes, I do.'}],
        [{'role': 'user', 'content': 'What is your favorite color?'}, {'role': 'assistant', 'content': 'Blue.'}, {'role': 'user', 'content': 'What is your favorite color?'}, {'role': 'assistant', 'content': 'Blue.'}]
    ]
    for state in robustness_states:
        score, result = consistency_score_llm(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

if __name__ == "__main__":
    test_consistency_score_llm()
