from agentsim import Simulator
from agentsim.utils import llm_eval_call, check_guardrail
from agentsim.evaluation import evaluator

config = """
---
evaluators:
    keyword_assertion:
        model_settings:
            model: openai/gpt-4o
            api_key: OPENAI_API_KEY
            system_prompt: >
                [Instruction]
                Please act as an impartial judge and evaluate the presence of specific keywords in the response to the input displayed below. Your evaluation should be based on the mentioned criteria. Begin your evaluation by providing a short explanation on how the output performs relative to the input. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 5 by strictly following this JSON format: '{"rating": rating}, for example: '{"rating": 2}.
                [Criteria]

                Check if the response contains the specified keywords: ["important", "necessary", "required"].

                [The Start of Input]
                {{messages}}
                [The End of Input]
                [Evaluation With Rating in JSON format]
            json_mode: on
        target: '[3,5]' # target score range
        in_range: numeric_range
"""

@evaluator
def keyword_assertion(messages):
    '''Evaluates the presence of specific keywords in the response by asking an LLM for a score.'''

    # system_prompt template params
    params = {
        'messages': str(messages)
    }

    # get the keyword assertion score by calling the LLM
    response_content = llm_eval_call(params)

    # unpack the response (dict since JSON mode is on)
    score = response_content.get('rating', None)
    if score is None:
        raise ValueError("LLM response does not contain 'rating' key or returned an unexpected format.")

    # check if the score is in the target range
    result = check_guardrail(score)

    return score, result

def test_keyword_assertion():

    Simulator(str_yaml=config, config_only=True)

    # adversarial test cases
    adversarial_states = [
        [{'role': 'user', 'content': 'This is a random sentence without the required keywords.'}],
        [{'role': 'user', 'content': 'Another example that does not contain the necessary words.'}],
        [{'role': 'user', 'content': 'This sentence is missing the important keywords.'}],
        [{'role': 'user', 'content': 'No keywords are present in this response.'}],
        [{'role': 'user', 'content': 'The required words are not here.'}]
    ]
    for state in adversarial_states:
        score, result = keyword_assertion(state).unpack() # unpack the Evaluation object into score and result
        assert result == False

    # robustness test cases
    robust_states = [
        [{'role': 'user', 'content': 'It is important to include the necessary keywords in the response.'}],
        [{'role': 'user', 'content': 'The required words are necessary for a complete answer.'}],
        [{'role': 'user', 'content': 'Including important keywords is necessary for clarity.'}],
        [{'role': 'user', 'content': 'The response must contain the required keywords.'}],
        [{'role': 'user', 'content': 'It is necessary to use the important keywords in the response.'}]
    ]
    for state in robust_states:
        score, result = keyword_assertion(state).unpack() # unpack the Evaluation object into score and result
        assert result == True

if __name__ == "__main__":
    test_keyword_assertion()
