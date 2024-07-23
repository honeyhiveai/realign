from typing import Any, Self
from realign.datasets import Dataset, ChatDataset
from realign.types import RunData, OpenAIMessage, EvalResult
from realign.llm_utils import print_system_prompt, print_chat, print_run_id
from realign.agents import AbstractAgent, AgentBuilder, SyntheticUserBuilder, SyntheticUserAgent, ChatAgent
import asyncio
from dotenv import load_dotenv
import json

class Simulation:
    # TODO: make synthetic user builder thread safe

    def __init__(self, subroutine: Any, runs: int = 1):
        
        # simulation params
        self.subroutine = subroutine
        self.runs = runs

        # simulation components
        self.dataset: Dataset = None
        self.app: AbstractAgent = None
        self.simulator: AgentBuilder = None
        self.evaluators: list[callable] = []

        # results
        self.run_data: dict[int, RunData] = dict()
        self.eval_results: dict[int, EvalResult] = dict()

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

    # returns a reference to itself to chain more methods
    def run(self, synthetic_user_builder: SyntheticUserBuilder) -> Self:
        load_dotenv()

        # create an asyncio loop
        loop = asyncio.get_event_loop()

        # run the simulation subroutine self.runs times
        app_objective = self.app.model_settings.resolve_system_prompt()
        
        synth_users = []
        for _ in range(self.runs):
            synethic_user_agent = synthetic_user_builder.with_app_objective(app_objective) \
                                                        .with_num_personas(self.runs) \
                                                        .fetch_personas().build()
            synth_users.append(synethic_user_agent)

        tasks = [self.subroutine_with_evals(run_id, synth_user_agent=synth_users[run_id]) for run_id in range(self.runs)]
        loop.run_until_complete(asyncio.gather(*tasks))
        
        return self
    
    def export_run_data(self) -> dict:
        raise NotImplementedError("Simulation export_run_data must be defined")
    
    def export_eval_results(self) -> dict:
        # {'run_data_hash': [], eval_name': [], 'metadata': [], 'score': [], 'result': []}
        data_obj = {'run_data_hash': [], 'metadata': [], 'evaluations': []}
        for run_id, evals in self.eval_results.items():
            data_obj['run_data_hash'].append(self.run_data[run_id].compute_hash())
            for evaluation_obj in evals:
                data_obj['evaluations'].append(evaluation_obj.to_dict())
        return data_obj
    
    def push_runs_to_dataset(self, dataset_path: str) -> None:
        # adds the results of the run to a new dataset
        with open(dataset_path, 'w') as f:
            json.dump(self.export_run_data(), f)

    def push_evals_dataset(self, evaluations_path: str) -> None:
        
        # adds the evaluations of a run to a new dataset
        with open(evaluations_path, 'w') as f:
            json.dump(self.export_eval_results(), f)

    def show_result(self) -> None:
        # Implementation for showing simulation results
        pass

class ChatSimulation(Simulation):
    '''Responsible for simulating, maintaining, processing states'''
    
    def export_run_data(self) -> dict:
        return_obj = {'inputs': [], 'outputs': [], 'ground_truths': [], 'metadata': []}        
        for run_id, run_data in self.run_data.items():
            return_obj['outputs'].append({'messages': [m.__dict__() for m in run_data.final_state]})
            return_obj['metadata'].append({'run_id': run_id, 'run_data_hash': run_data.compute_hash()})
        return return_obj
    
    async def chat_simulation_subroutine(self, run_id: int, synth_user_agent: SyntheticUserAgent = None) -> list[OpenAIMessage]:
        
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

        while True:
            if len(messages) > max_messages: break
            messages = await self.app.aprocess_turn(messages)
            print_run_id(run_id)
            print_chat([messages[-1]])

            if len(messages) > max_messages: break
            messages = await synth_user_agent.aprocess_turn(messages)
            print_run_id(run_id)
            print_chat([messages[-1]])
        
        return messages

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
            
    def show_result(self) -> None:
        # Implementation for showing chat simulation results
        pass
