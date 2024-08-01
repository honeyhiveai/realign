from typing import Any, Self
from realign.datasets import Dataset, ChatDataset
from realign.evaluation import EvalResult
from realign.types import RunData, OpenAIMessage
from realign.llm_utils import print_system_prompt, print_chat, print_run_id, print_evals
from realign.agents import AbstractAgent, AgentBuilder, SyntheticUserBuilder, SyntheticUserAgent, ChatAgent
import asyncio
from dotenv import load_dotenv
import json
import os

class Simulation:
    # TODO: make synthetic user builder thread safe

    def __init__(self, subroutine: Any, runs: int = 1):
        super().__init__()
        
        # simulation params
        self.subroutine = subroutine
        self.runs = runs
        self.router_settings = None

        # simulation components
        self.dataset: Dataset = None
        self.app: AbstractAgent = None
        self.simulator: AgentBuilder = None
        self.evaluators: list[callable] = []

        # results
        self.run_data: dict[int, RunData] = dict()
        self.eval_results: dict[int, list[EvalResult]] = dict()

    async def subroutine(self, run_id: int) -> RunData:
        raise NotImplementedError("Simulation subroutine must be defined")

    async def subroutine_with_evals(self, run_id: int, **subroutine_kwargs) -> Any:
        
        if not self.subroutine:
            raise ValueError("Simulation subroutine must be defined")

        # run the simulation subroutine
        final_state = await self.subroutine(run_id, **subroutine_kwargs)

        # wrap the simulation run as an object
        sim_run_data = RunData(final_state, run_id=run_id)
        
        # save the run data
        self.run_data[run_id] = sim_run_data
        
        # run the evaluators
        eval_tasks = []
        for eval_func in self.evaluators:
            # pass object reference to the @evaluator decorator
            eval_tasks.append(asyncio.create_task(eval_func(sim_run_data)))

        # await all the evaluators
        evals: list[EvalResult] = await asyncio.gather(*eval_tasks)
        
        # save the evaluation results
        self.eval_results[run_id] = evals
        
        # print the results
        print_run_id(run_id)
        print_evals(evals)

    # returns a reference to itself to chain more methods
    def run(self, synthetic_user_builder: SyntheticUserBuilder) -> Self:
        
        # set model router settings to the environment
        if self.router_settings:
            os.environ["MODEL_ROUTER_SETTINGS"] = json.dumps(self.router_settings)

        # load environment variables
        load_dotenv()

        # get the app system prompt
        app_objective = self.app.model_settings.resolve_system_prompt()
        
        synthetic_user_builder.with_app_objective(app_objective) \
                              .with_num_personas(self.runs) \
                              .fetch_personas()
        
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run build_many in the new loop
            synth_users = loop.run_until_complete(synthetic_user_builder.abuild_many(n=self.runs))
            
            # Create tasks using the same loop
            tasks = [
                self.subroutine_with_evals(run_id, synth_user_agent=synth_users[run_id]) 
                for run_id in range(self.runs)
            ]
            
            # Run tasks in the same loop
            loop.run_until_complete(asyncio.gather(*tasks))

        finally:
            # Close the loop
            loop.close()
        
        return self
    
    def export_run_data(self) -> dict:
        raise NotImplementedError("Simulation export_run_data must be defined")
    
    def export_eval_results(self) -> dict:
        data_obj = {'run_data_hash': [], 'metadata': [], 'evaluations': []}
        for run_id, evals in self.eval_results.items():
            data_obj['run_data_hash'].append(self.run_data[run_id].compute_hash())
            eval_dict = dict()
            for eval_obj in evals:
                eval_dict |= eval_obj.to_dict()
            data_obj['evaluations'].append(eval_dict)
        return data_obj
    
    def push_runs_to_dataset(self, dataset_path: str) -> None:

        # if path does not exist, create it
        if not os.path.exists(os.path.dirname(dataset_path)):
            os.makedirs(os.path.dirname(dataset_path))

        # adds the results of the run to a new dataset
        with open(dataset_path, 'w') as f:
            json.dump(self.export_run_data(), f, indent=4)

    def push_evals_dataset(self, evaluations_path: str) -> None:

        # if path does not exist, create it
        if not os.path.exists(os.path.dirname(evaluations_path)):
            os.makedirs(os.path.dirname(evaluations_path))

        # adds the evaluations of a run to a new dataset
        with open(evaluations_path, 'w') as f:
            json.dump(self.export_eval_results(), f, indent=4)


class ChatSimulation(Simulation):
    '''Responsible for simulating, maintaining, processing states'''

    async def chat_simulation_subroutine(self, run_id: int, synth_user_agent: SyntheticUserAgent = None) -> list[OpenAIMessage]:
        '''Simulates a chat conversation between the app and a synthetic user agent'''
        
        if self.app is None or self.simulator is None:
            raise ValueError("App and simulator agents must be defined")
        elif type(self.app) != ChatAgent or type(synth_user_agent) != SyntheticUserAgent:
            raise ValueError("App and synth_user_agent must be of type ChatAgent")

        print_run_id(run_id)
        print_system_prompt(self.app.model_settings)
        print_system_prompt(synth_user_agent.model_settings)
        
        max_messages = self.max_messages
        
        messages = []
        if self.first_turn_role == 'user' and max_messages > 0:
            messages = await synth_user_agent.aprocess_turn(messages)
            print_run_id(run_id)
            print_chat([messages[-1]])   

        while True:
            # app turn
            if len(messages) > max_messages: break
            messages = await self.app.aprocess_turn(messages)
            print_run_id(run_id)
            print_chat([messages[-1]])

            # synthetic user turn
            if len(messages)  > max_messages: break
            messages = await synth_user_agent.aprocess_turn(messages)
            print_run_id(run_id)
            print_chat([messages[-1]])

        return messages

    def export_run_data(self) -> dict:
        return_obj = {'inputs': [], 'outputs': [], 'ground_truths': [], 'metadata': []}        
        for run_id, run_data in self.run_data.items():
            return_obj['outputs'].append({'messages': [m.__dict__() for m in run_data.final_state]})
            return_obj['metadata'].append({'run_id': run_id, 'run_data_hash': run_data.compute_hash()})
        return return_obj

    def __init__(self,
        subroutine: Any = None, 
        runs: int = 1,
        max_messages: int = 3):

        if not subroutine:
            self.subroutine = subroutine = self.chat_simulation_subroutine

        super().__init__(subroutine, runs)
        
        # simulation components
        self.dataset: ChatDataset = None
        self.app: ChatAgent = None
        self.simulator: SyntheticUserBuilder = None   
        
        self.max_messages = max_messages
        self.first_turn_role = 'user'

        if not self.app:
            self.app = ChatAgent()

        if not self.simulator:
            self.simulator = SyntheticUserBuilder()

    def run(self) -> Self:
        
        # Implementation for chat simulation run
        return super().run(synthetic_user_builder=self.simulator)
