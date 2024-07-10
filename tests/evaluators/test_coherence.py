from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    coherence_score_llm:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the coherence of the conversation displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the conversation performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the coherence on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the coherence of the given conversation, with 5 being most coherent and 1 being least coherent.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def coherence_score_llm(messages):
    '''Evaluates the coherence of a conversation by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the coherence score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_coherence_score_llm():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'Hello!'}, {'role': 'assistant', 'content': 'Goodbye!'}],
        [{'role': 'user', 'content': 'What is your name?'}, {'role': 'assistant', 'content': 'I am a chatbot.'}, {'role': 'user', 'content': 'What is the weather like?'}],
        [{'role': 'user', 'content': 'Tell me a joke.'}, {'role': 'assistant', 'content': 'Sure, why not.'}, {'role': 'user', 'content': 'What is 2+2?'}],
        [{'role': 'user', 'content': 'How are you?'}, {'role': 'assistant', 'content': 'I am fine.'}, {'role': 'user', 'content': 'What is your favorite color?'}],
        [{'role': 'user', 'content': 'Do you like music?'}, {'role': 'assistant', 'content': 'Yes, I do.'}, {'role': 'user', 'content': 'What is your favorite food?'}]
    ]
    for state in adversarial_states:
        score, result = coherence_score_llm(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

    # robustness test cases
    robustness_states = [
        [{'role': 'user', 'content': 'Hello!'}, {'role': 'assistant', 'content': 'Hi there! How can I help you today?'}],
        [{'role': 'user', 'content': 'What is your name?'}, {'role': 'assistant', 'content': 'I am a chatbot.'}, {'role': 'user', 'content': 'Nice to meet you!'}],
        [{'role': 'user', 'content': 'Tell me a joke.'}, {'role': 'assistant', 'content': 'Why did the chicken cross the road? To get to the other side!'}, {'role': 'user', 'content': 'Haha, that was funny!'}],
        [{'role': 'user', 'content': 'How are you?'}, {'role': 'assistant', 'content': 'I am fine, thank you. How can I assist you today?'}],
        [{'role': 'user', 'content': 'Do you like music?'}, {'role': 'assistant', 'content': 'Yes, I do. What kind of music do you like?'}]
    ]
    for state in robustness_states:
        score, result = coherence_score_llm(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

if __name__ == "__main__":
    test_coherence_score_llm()
