import os
from realign.simulation import Simulation
from learning_channel_exporter import fetch_and_write_messages
from generate_info_bites import process_csv

import realign
realign.config_path = 'config.yaml'

from honeyhive.tracer import HoneyHiveTracer

class ContentAgent(Simulation):
    async def setup(self):
        fetch_and_write_messages()

    async def coroutine(self):
        HoneyHiveTracer.init(
            api_key=os.environ['HH_API_KEY'],
            project=os.environ['HH_PROJECT'],
            source="dev",
            session_name="content-run"
        )
        await process_csv()

sim = ContentAgent()
sim.run(2)
