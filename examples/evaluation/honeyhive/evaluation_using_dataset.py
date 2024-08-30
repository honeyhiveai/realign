
from realign.evaluation import Evaluation
from realign.llm_utils import allm_messages_call, llm_messages_call

import realign
realign.config.path = 'examples/evaluation/honeyhive/config.yaml' 
realign.tracing.honeyhive_key = '<YOUR_HONEYHIVE_API_KEY>'
realign.tracing.honeyhive_project = '<YOUR_HONEYHIVE_PROJECT_NAME>'


class SampleDatasetEvaluation(Evaluation):

    async def main(self, run_context):
        # inputs -> datapoint as dict
        inputs = run_context.inputs 

        ### Evaluation body starts here ###

        message = llm_messages_call(
            agent_name='summary_writer', 
            template_params=inputs,
        )

        ### Evaluation body ends here ###
        
        # Return output to stitch to the session
        return message.content


eval = SampleDatasetEvaluation(
    evaluation_name='<EVALUATION NAME>', 
    dataset_id='<DATASET ID>'
)
eval.run()
