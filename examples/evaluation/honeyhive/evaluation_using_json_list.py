
from realign.evaluation import Evaluation
from realign.llm_utils import llm_messages_call

import realign
realign.config.path = 'examples/evaluation/honeyhive/config.yaml' 
realign.tracing.honeyhive_key = '<YOUR_HONEYHIVE_API_KEY>'
realign.tracing.honeyhive_project = '<YOUR_HONEYHIVE_PROJECT_NAME>'

class SampleEvaluation(Evaluation):

    async def main(self, run_context):
        # inputs -> json as dict
        inputs = run_context.inputs 

        ### Evaluation body starts here ###

        message = llm_messages_call(
            agent_name='summary_writer', 
            template_params=inputs,
        )

        ### Evaluation body ends here ###
        
        # Return output to stitch to the session
        return message.content

sample_json_list = [
    {
        "product_type": "electric vehicles",
        "region": "western europe",
        "time_period": "first half of 2023",
        "metric_1": "total revenue",
        "metric_2": "market share"
    },
    {
        "product_type": "gaming consoles",
        "region": "north america",
        "time_period": "holiday season 2022",
        "metric_1": "units sold",
        "metric_2": "gross profit margin"
    },
    {
        "product_type": "smart home devices",
        "region": "australia and new zealand",
        "time_period": "fiscal year 2022-2023",
        "metric_1": "customer acquisition cost",
        "metric_2": "average revenue per user"
    }
]

eval = SampleEvaluation(
    evaluation_name='Sample Evaluation', 
    query_list=sample_json_list

)
eval.run()
