from realign import evaluator, aevaluator, allm_messages_call, Simulation

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

    async def main(self, run_context) -> None:
        
        # write tweet
        message = await allm_messages_call(
                            agent_name='twitter_content_agent', 
                            messages=[{
                                'role': 'user', 
                                'content': self.tweet_prompt
                            }],
                        )
        
        return message.content
    
    async def windup(self):
        await super().windup()
        
        elo_ratings = await aevaluator.llm_tweet_choice_judge(self.final_states)
        
        # print the best and worst tweets
        print('-'*100)
        print(f'Best tweet with rating {elo_ratings[0][1]}:\n\n', elo_ratings[0][0])
        print('-'*100)
        print(f'Worst tweet with rating {elo_ratings[-1][1]}:\n\n', elo_ratings[-1][0])
        print('-'*100)


sim = TweetBot('write an insightful tweet based on the abstract of the research paper')
sim.run(3)
