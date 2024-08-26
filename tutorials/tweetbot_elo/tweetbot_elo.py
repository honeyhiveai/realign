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
        
        print(run_context.run_id)
        print(message.content)
        print('\n\n\n')
        
        return message.content
    
    async def windup(self):
        
        elo_ratings = await aevaluator['llm_tweet_choice_judge'](self.final_states)
        
        # print the best and worst tweets
        print(f'\n\nBest tweet with rating {elo_ratings[0][1]}:\n', elo_ratings[0][0])
        
        print(f'\n\nWorst tweet with rating {elo_ratings[-1][1]}:\n', elo_ratings[-1][0])
        
        return await super().windup()


sim = TweetBot(
    """
    This paper titled 'LLM experiments with simulation: Large Language Model Multi-Agent System for Simulation Model Parametrization in Digital Twins' presents a novel design of a multi-agent system framework that applies large language models (LLMs) to automate the parametrization of simulation models in digital twins. This framework features specialized LLM agents tasked with observing, reasoning, decision-making, and summarizing, enabling them to dynamically interact with digital twin simulations to explore parametrization possibilities and determine feasible parameter settings to achieve an objective. The proposed approach enhances the usability of simulation model by infusing it with knowledge heuristics from LLM and enables autonomous search for feasible parametrization to solve a user task. Furthermore, the system has the potential to increase user-friendliness and reduce the cognitive load on human users by assisting in complex decision-making processes.
    """
    )
sim.run(5)
