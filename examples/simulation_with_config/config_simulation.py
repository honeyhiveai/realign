from realign.simulation import Simulation
from realign.llm_utils import allm_messages_call

import realign
realign.config.path = 'examples/simulation_with_config/config.yaml'

class TestSimulation(Simulation):
    
    async def setup(self):
        
        message = await allm_messages_call(
                agent_name='prompt_writer', 
                template_params={
                    'user_request': 'Write a post on the negative impacts of AI',
                    'platform': 'Twitter',
                },
            )
        self.tweet_prompt = message.content
        print('tweet_prompt', self.tweet_prompt)
        
        # self.evaluators = [evaluator(len)]
        self.tweet_prompt = 'Write a post on the negative impacts of AI'

    async def main(self, run_context) -> None:
        
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


sim = TestSimulation()
sim.run(10)
