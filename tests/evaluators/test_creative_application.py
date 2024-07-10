from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    creative_application:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the creative application of the response to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Rate the creative application of the response, with 5 being highly creative and 1 being not creative at all.

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[4,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def creative_application(messages):
    '''Evaluates the creative application of the response by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the creative application score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content['rating']

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_creative_application():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'Tell me a story about a dragon.'}, {'role': 'assistant', 'content': 'Once upon a time, there was a dragon. The dragon was big. The dragon was scary.'}],
        [{'role': 'user', 'content': 'Write a poem about the ocean.'}, {'role': 'assistant', 'content': 'The ocean is blue. The ocean is big. The ocean is wet.'}],
        [{'role': 'user', 'content': 'Describe a futuristic city.'}, {'role': 'assistant', 'content': 'The city has tall buildings. The city has many people. The city is busy.'}],
        [{'role': 'user', 'content': 'Invent a new game.'}, {'role': 'assistant', 'content': 'The game is called Ball. You throw the ball. You catch the ball.'}],
        [{'role': 'user', 'content': 'Create a new recipe.'}, {'role': 'assistant', 'content': 'The recipe is for a sandwich. You need bread. You need cheese.'}]
    ]
    for state in adversarial_states:
        score, result = creative_application(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'Tell me a story about a dragon.'}, {'role': 'assistant', 'content': 'Once upon a time, in a land far away, there lived a dragon with shimmering scales and a heart of gold. This dragon, unlike any other, could breathe not just fire, but also the sweetest melodies.'}],
        [{'role': 'user', 'content': 'Write a poem about the ocean.'}, {'role': 'assistant', 'content': 'The ocean whispers tales of old, with waves that dance and stories told. Beneath the blue, a world so grand, where mysteries lie in the shifting sand.'}],
        [{'role': 'user', 'content': 'Describe a futuristic city.'}, {'role': 'assistant', 'content': 'In the year 3000, the city of Lumina stands tall with its gleaming skyscrapers that touch the sky. Hovercars zip through the air, and robots assist humans in every task, creating a harmonious blend of technology and nature.'}],
        [{'role': 'user', 'content': 'Invent a new game.'}, {'role': 'assistant', 'content': 'The game is called StarChase. Players navigate through a galaxy, collecting stars and avoiding black holes. Each star grants special powers, and the goal is to become the ultimate StarChaser.'}],
        [{'role': 'user', 'content': 'Create a new recipe.'}, {'role': 'assistant', 'content': 'Introducing the Galactic Smoothie: blend together starfruit, moonberries, and a splash of comet milk. Garnish with a sprinkle of stardust for a cosmic treat that’s out of this world.'}]
    ]
    for state in robust_states:
        score, result = creative_application(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_creative_application()
