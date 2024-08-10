from realign.llm_utils import State
from realign.simulation import Simulation
from learning_channel_exporter import fetch_and_write_messages
from generate_info_bites import process_csv
import os

class TutorSimulation(Simulation):
    async def setup(self, runs):
        fetch_and_write_messages()

    async def coroutine(self, run_id):
        # check if this file exists
        if os.path.exists("processed_links.txt"):
            os.remove("processed_links.txt")
        await process_csv()
        return ""

sim = TutorSimulation()
sim.run(3)
