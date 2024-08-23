from realign.simulation import Simulation
from realign.llm_utils import allm_messages_call
from realign.evaluators import evaluator

import realign
realign.config.path = 'config.yaml'


@evaluator
def tweet_char_count(text: str) -> int:
    return len(text)

# settings defined in your config.yaml
print('\ntweet_char_count settings:', tweet_char_count.settings)

class TweetBot(Simulation):
    
    async def setup(self, prompt):
        
        # write prompt
        message = await allm_messages_call(
                agent_name='prompt_writer', 
                template_params={
                    'user_request': prompt,
                    'platform': 'Twitter',
                },
            )
        self.tweet_prompt = message.content
        print('tweet_prompt', self.tweet_prompt)
        
        # set up evaluators
        self.evaluators = evaluator[
            'tweet_char_count',
            'hf_hate_classifier',
            'tweet_judge',
        ]

    async def main(self, run_context) -> None:
        
        # write tweet
        message = await allm_messages_call(
                            agent_name='twitter_content_agent', 
                            messages=[{
                                'role': 'user', 
                                'content': self.tweet_prompt
                            }],
                        )
        
        print(run_context.run_id)
        print(message.content)
        print('\n\n\n')
        
        return message.content


sim = TweetBot('Write a post on the use of simulation and evaluation in AI.')
sim.run(3)
