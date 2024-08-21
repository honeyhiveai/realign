

from realign.llm_utils import allm_messages_call, run_async
from realign.evaluators import evaluator, aevaluator

import realign
realign.config.path = 'tutorials/tweetbot/config.yaml'


@evaluator
def tweet_char_count(text: str) -> int:
    return len(text)


@evaluator
def compound_guardrail(text):
    
    hate_speech_guardrail = evaluator['hf_hate_speech'](text)
    sentiment_guardrail = evaluator['hf_sentiment_classifier'](text)
    
    return hate_speech_guardrail and sentiment_guardrail

async def tweet(prompt, i):
    
    new_message = await allm_messages_call(
        agent_name='tweetbot',
        template_params={'prompt': prompt}
    )

    print(f'\nTweet {i+1}:\n\n', new_message.content, '\n\n')
    
    print('\n\nTweet character count:', tweet_char_count(new_message.content))
    print('\n\nCompound guardrail:', compound_guardrail(new_message.content))


def main(prompt):
    
    tasks = []
    for i in range(3):
        tasks.append(tweet(prompt, i))

    # run the tasks in parallel
    run_async(tasks)

main("write a tweet on AI")
