from realign.simulation import Simulation
from realign.llm_utils import allm_messages_call

# Place the code below at the beginning of your application to initialize the tracer


import realign
realign.config_path = 'examples/content_agent/config.yaml'


class TestSimulation(Simulation):

    async def coroutine(self):

        message = await allm_messages_call('twitter_content_agent', 
                                           messages=[{
                                                'role': 'user', 
                                                'content': 'using synthetic datasets to tune evaluators'
                                            }],
                                        )
        print(message.content)

sim = TestSimulation()
sim.run(10)