
from realign.simulation import Simulation
from realign.llm_utils import allm_messages_call, llm_messages_call
from realign.tracing import get_tracer

import honeyhive
from honeyhive.models import components, operations
from honeyhive.tracer import HoneyHiveTracer
import os
import realign


class Evaluation(Simulation):

    def __init__(self, 
                 evaluation_name, 
                 dataset_id = None, 
                 query_list = None):
        super().__init__()
        self.hhai = honeyhive.HoneyHive(
            bearer_auth=os.environ['HH_API_KEY'],
        )
        self.eval_name = evaluation_name
        self.dataset_id = dataset_id
        self.evaluation_session_ids = []

        if self.dataset_id:
            dataset = self.hhai.datasets.get_datasets(
                project= os.environ['HH_PROJECT'],
                dataset_id= dataset_id,
            )
            if dataset.object.testcases is not None and len(dataset.object.testcases) > 0:
                self.dataset = dataset.object.testcases[0]
                print(self.dataset.datapoints)
            else:
                raise RuntimeError("No dataset found with id - {} for project - {}".format(id, os.environ['HH_PROJECT'])) 
            self.runs = len(self.dataset.datapoints)
        else:
            self.dataset = None
        self.query_list = query_list
        self.instrument_manual_tracing = True 


    async def setup(self):

        eval_run = self.hhai.runs.create_run(request=components.CreateRunRequest(
            project=os.environ['HH_PROJECT'],
            name=self.eval_name,
            dataset_id=self.dataset_id,
            event_ids=[],
        ))
        self.eval_run = eval_run.create_run_response
        

    async def main(self, run_context) -> None:

        inputs = None
        run_unique_iterator = run_context.run_id

        if self.dataset and self.dataset.datapoints and len(self.dataset.datapoints) > 0 :
            try:
                datapoint_id = self.dataset.datapoints[run_unique_iterator]
                datapoint_response = self.hhai.datapoints.get_datapoint(id = datapoint_id)
                datapoint = datapoint_response.object.datapoint[0]
                inputs = datapoint.inputs
                
            except Exception as e:
                print(e)
            
            
        elif self.query_list:
            inputs = self.query_list[run_unique_iterator]

        tracer = get_tracer('evaluation')
        if not tracer:
            raise Exception(f"Unable to initiate Honeyhive Tracer. Cannot run Evaluation")
        
        tracer.initialize_trace(self.eval_name)

        evaluation_output = None
        try:
            evaluation_output = await self.eval_function(inputs)
        except Exception as error:
            print(error)
            pass

        try:
            tracing_metadata = { 
                "run_id": self.eval_run.run_id,
                "inputs": inputs 
            }
            if datapoint:
                tracing_metadata["datapoint_id"] = datapoint_id
                tracing_metadata["dataset_id"] = self.dataset_id
            if evaluation_output:
                tracing_metadata["outputs"] = evaluation_output

            tracer.add_trace_metadata(tracing_metadata)
        except Exception as e:
            print(e)
        self.evaluation_session_ids.append(HoneyHiveTracer.session_id)

        return 
    
    async def windup(self):
        self.hhai.runs.update_run(
            run_id = self.eval_run.run_id,
            update_run_request=components.UpdateRunRequest(
                event_ids = self.evaluation_session_ids,
                status = "completed"
            )
        )
        return await super().windup()

    
    async def eval_function(self, inputs_json):
        pass
    