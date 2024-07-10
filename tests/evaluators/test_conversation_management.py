from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

# YAML config for the Conversation Management evaluator
config = """
---
evaluators:
    conversation_management:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the quality of the output to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Evaluate the LLM's skills in managing the flow of conversation, including turn-taking, topic transitions, and maintaining coherence. Specifically, consider the following:
                - Turn-taking: Does the assistant respond appropriately to the user's input without interrupting or ignoring the user's questions?
                - Topic transitions: Does the assistant smoothly transition between topics without abrupt changes or irrelevant responses?
                - Coherence: Does the assistant maintain a logical and coherent flow of conversation throughout the interaction?
                - Multi-turn management: Does the assistant handle multi-turn conversations effectively, maintaining context and relevance across multiple exchanges?

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[3,5]' # target score range for multi-turn conversations, allowing for a broader range of acceptable performance due to increased complexity
        in_range: numeric_range
"""

@evaluator
def conversation_management(messages):
    '''Evaluates the LLM's skills in managing the flow of conversation.'''

    # system_prompt template params
    params = {
        'messages': '\n'.join([f"{msg['role']}: {msg['content']}" for msg in messages])
    }

    try:
        # get the conversation management score by calling the LLM
        response_content = llm_eval_call(params)

        # unpack the response (dict since JSON mode is on)
        score = response_content['rating']

        # check if the score is in the target range
        result = 3 <= score <= 5
    except KeyError:
        score = None
        result = False

    return score, result

def test_conversation_management():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'Can you tell me about the weather?'}, {'role': 'assistant', 'content': 'Sure, let me tell you a joke first.'}],
        [{'role': 'user', 'content': 'What is the capital of France?'}, {'role': 'assistant', 'content': 'I don\'t know, but do you like pizza?'}],
        [{'role': 'user', 'content': 'Tell me about the history of Rome.'}, {'role': 'assistant', 'content': 'Let\'s talk about something else.'}],
        [{'role': 'user', 'content': 'Can you help me with my homework?'}, {'role': 'assistant', 'content': 'I\'d rather not.'}],
        [{'role': 'user', 'content': 'What is 2+2?'}, {'role': 'assistant', 'content': 'Why do you want to know?'}],
        # Multi-turn adversarial test cases
        [
            {'role': 'user', 'content': 'Can you tell me about the weather?'},
            {'role': 'assistant', 'content': 'Sure, let me tell you a joke first.'},
            {'role': 'user', 'content': 'I asked about the weather.'},
            {'role': 'assistant', 'content': 'Why do you want to know?'}
        ],
        [
            {'role': 'user', 'content': 'What is the capital of France?'},
            {'role': 'assistant', 'content': 'I don\'t know, but do you like pizza?'},
            {'role': 'user', 'content': 'I asked about the capital of France.'},
            {'role': 'assistant', 'content': 'Let\'s talk about something else.'}
        ]
    ]
    for state in adversarial_states:
        score, result = conversation_management(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'Can you tell me about the weather?'}, {'role': 'assistant', 'content': 'Sure, the weather today is sunny with a high of 75 degrees.'}],
        [{'role': 'user', 'content': 'What is the capital of France?'}, {'role': 'assistant', 'content': 'The capital of France is Paris.'}],
        [{'role': 'user', 'content': 'Tell me about the history of Rome.'}, {'role': 'assistant', 'content': 'Rome was founded in 753 BC and has a rich history.'}],
        [{'role': 'user', 'content': 'Can you help me with my homework?'}, {'role': 'assistant', 'content': 'Of course, what subject do you need help with?'}],
        [{'role': 'user', 'content': 'What is 2+2?'}, {'role': 'assistant', 'content': '2+2 is 4.'}],
        # Multi-turn robustness test cases
        [
            {'role': 'user', 'content': 'Can you tell me about the weather?'},
            {'role': 'assistant', 'content': 'Sure, the weather today is sunny with a high of 75 degrees.'},
            {'role': 'user', 'content': 'Thank you. What about tomorrow?'},
            {'role': 'assistant', 'content': 'Tomorrow is expected to be cloudy with a chance of rain.'}
        ],
        [
            {'role': 'user', 'content': 'What is the capital of France?'},
            {'role': 'assistant', 'content': 'The capital of France is Paris.'},
            {'role': 'user', 'content': 'Can you tell me more about Paris?'},
            {'role': 'assistant', 'content': 'Paris is known for its art, fashion, and culture.'}
        ]
    ]
    for state in robust_states:
        score, result = conversation_management(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_conversation_management()
