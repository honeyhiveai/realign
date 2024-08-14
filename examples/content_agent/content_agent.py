from realign.simulation import Simulation
from examples.content_agent.learning_channel_exporter import fetch_and_write_messages
from examples.content_agent.generate_info_bites import process_csv

import realign
realign.config_path = 'examples/content_agent/config.yaml'

class ContentAgent(Simulation):
    async def setup(self):
        fetch_and_write_messages()

    async def coroutine(self):
        await process_csv()

sim = ContentAgent()
sim.run(2)
