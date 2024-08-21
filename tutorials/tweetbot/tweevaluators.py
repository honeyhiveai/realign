from realign.llm_utils import allm_messages_call, run_async
from realign.evaluators import aevaluator

import realign
realign.config_path = 'tutorials/tweetbot/config.yaml'
import asyncio

@aevaluator
async def tweet_quality_eval(text, criteria=None):
    return await aevaluator['llm_rating_json'](criteria, text)

async def tweet():
    new_message = await allm_messages_call(
        agent_name='tweetbot',
        template_params={'prompt': 'make a post on AI'}
    )
    print(new_message.content)
    
    quality = await tweet_quality_eval('new_message.content')
    print(quality)

asyncio.run(tweet())
# print(tweet_quality_eval.prev_run)

# tweet = '''
# 🔍 Want to master prompt engineering? Start by being specific! Clear, detailed prompts yield better results. Experiment with different phrasings and provide context to guide the AI. Remember, iteration is key—refine and test to get the best output! 💡 #AI #PromptEngineering'''


# print(evaluator['llm_rating_json'].prev_run)

# print(tweet_char_count(tweet))

# print(quality_score(tweet))
# print(quality_score.prev_run)




# print(evaluator['hf_hate_speech'](tweet))
# print(evaluator['hf_hate_speech'].prev_run)

# print(compound_guardrail(tweet))
# print(compound_guardrail.prev_run)

# print(evaluator['hf_pipeline'](tweet, 'text-classification', model="facebook/roberta-hate-speech-dynabench-r4-target"))
