import realign 
realign.tracing.honeyhive_key = 'YmsydjFzdHB5cWQ4aHN2cjB2cTll'
realign.tracing.honeyhive_project = 'Putnam'

from realign import Evaluation, Simulation, router, allm_messages_call

router.model_router_settings = {
    'openai/*': {
        'batch_size': 100,
        'requests_per_minute': 3
    }
}

class SampleEvaluation(Evaluation):
    async def main(self, run_context):
        # inputs -> datapoint as dict
        inputs = run_context.inputs 

        ### Evaluation body starts here ###
        inputs = "Solve the Riemann hypothesis"
        
        # TODO How to blast 10 times
        message = await allm_messages_call(
            agent_name='putnam_solver', 
            messages=[
                { "role": "user", "content": inputs }
            ]
        )
        
        ### Evaluation body ends here ###

        # Return output to stitch to the session
        return message.content

eval = SampleEvaluation(
    evaluation_name='Sample Evaluation', 
    dataset_id='66e3439452844594dd6316d0'
)
eval.run(1)

class SampleSimulation(Simulation):
    async def main(self, run_context):
        message = await allm_messages_call(
            agent_name='putnam_solver', 
            messages=[
                { "role": "user", "content": "Solve the Riemann hypothesis" }
            ]
        )
        return message.content

sim = SampleSimulation()
sim.run(1)