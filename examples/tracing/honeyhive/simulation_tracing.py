import os
from realign.simulation import Simulation
from realign.llm_utils import allm_messages_call, llm_messages_call

import realign
realign.config.path = 'examples/tracing/honeyhive/config.yaml'
realign.tracing.honeyhive_key = '<YOUR_HONEYHIVE_API_KEY>'
realign.tracing.honeyhive_project = '<YOUR_HONEYHIVE_PROJECT_NAME>'


class SimulationAgent(Simulation):

    async def main(self, run_context):

        message_1 = llm_messages_call(
            agent_name='writer_1', 
        )

        message_2 = llm_messages_call(
            agent_name='writer_2', 
        )

        return message_2.content

sim = SimulationAgent()
sim.run(5)
