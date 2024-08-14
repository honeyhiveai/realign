from realign.datasets import Dataset, ChatDataset
from realign.evaluation import EvalResult
from realign.types import RunData, OpenAIMessage
from realign.llm_utils import router, State, system_prompt_str, str_msgs, print_run_id, print_evals
from realign.agents import AbstractAgent, AgentBuilder, SyntheticUserFactory, SyntheticUserAgent, ChatAgent

from typing import Any, Self, Callable, Coroutine
import asyncio
from dotenv import load_dotenv
import json
import os
import inspect

class Simulation:
    # TODO: make synthetic user builder thread safe

    def __init__(self):
        self.runs = None
        
        # if provided, set the router settings to honor rate limits
        self.router_settings = None
        
        # evaluators
        self.evaluators = []

        # results
        self.run_data: dict[int, RunData] = dict()
        self.eval_results: dict[int, list[EvalResult]] = dict()

    async def setup(self, runs: int):
        '''Sets up objects used in the simulation'''

        # simulation components accessible to the coroutine
        self.dataset: Dataset = None
        self.app: AbstractAgent = None
        self.evaluators: list[Callable] = []

    async def coroutine(self, run_id: int) -> RunData:
        raise NotImplementedError("Simulation coroutine must be defined")

    async def coroutine_with_evals(self, run_id: int) -> Any:

        # run the simulation coroutine
        if inspect.getfullargspec(self.coroutine).args == ['self']:
            final_state = await self.coroutine()
        else:
            final_state = await self.coroutine(run_id)

        # wrap the simulation run as an object
        sim_run_data = RunData(final_state, run_id=run_id)
        
        # save the run data
        self.run_data[run_id] = sim_run_data
        
        # print the eval results
        if self.evaluators and len(self.evaluators) > 0:

            # run the evaluators  
            eval_tasks = []
            for eval_func in self.evaluators:
                # pass object reference to the @evaluator decorator
                eval_tasks.append(asyncio.create_task(eval_func(sim_run_data)))

            # await all the evaluators
            evals: list[EvalResult] = await asyncio.gather(*eval_tasks)
            
            # save the evaluation results
            self.eval_results[run_id] = evals

            print_run_id(run_id)
            print_evals(evals)
            
    async def run_simulation(self):
        
        # start timer
        start = asyncio.get_event_loop().time()
        
        try:
            
            # call setup
            if inspect.getfullargspec(self.setup).args == ['self']:
                await self.setup()
            else:
                await self.setup(runs=self.runs)
            
            simulation_run_tasks = [
                self.coroutine_with_evals(run_id)
                for run_id in range(self.runs)
            ]
            
            await asyncio.gather(*simulation_run_tasks)

        finally:
            if router:
                await router.shutdown()
                router_stats = router.get_stats()
            else:
                router_stats = None

            # end timer
            end = asyncio.get_event_loop().time()

            print('\n\n' + '-'*100)
            print('Simulation Stats:')
            print(f'\tSimulation Duration: {(end - start):.3f} sec')
            print(f'\tRuns: {self.runs} runs')

            print('-'*100)
            for model_name, model_stats in router_stats.items():
                print(f'{model_name} Stats:')
                print(json.dumps(model_stats, indent=4).replace('"', ''))
            
            print('-'*100 + '\n\n')

    def run(self, runs: int = 3) -> Self:
        '''Runs the main event loop for the simulation'''

        # simulation is fundamentally a coroutine that runs N times
        self.runs = self.runs or runs
        
        # set model router settings to the environment
        if self.router_settings:
            os.environ["MODEL_ROUTER_SETTINGS"] = json.dumps(self.router_settings)

        # load environment variables
        load_dotenv()
        
        try:
            asyncio.run(self.run_simulation())
        except KeyboardInterrupt:
            print("Simulation interrupted by user")
        except Exception as e:
            print(f"An error occurred during simulation: {e}")
            raise
        
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
    
    def push_runs_to_dataset(self, dataset_path: str = 'data/run_data.json') -> None:

        # if path does not exist, create it
        if not os.path.exists(os.path.dirname(dataset_path)):
            os.makedirs(os.path.dirname(dataset_path))

        # adds the results of the run to a new dataset
        with open(dataset_path, 'w') as f:
            json.dump(self.export_run_data(), f, indent=4)

    def push_evals_dataset(self, evaluations_path: str = 'data/eval_data.json') -> None:

        # if path does not exist, create it
        if not os.path.exists(os.path.dirname(evaluations_path)):
            os.makedirs(os.path.dirname(evaluations_path))

        # adds the evaluations of a run to a new dataset
        with open(evaluations_path, 'w') as f:
            json.dump(self.export_eval_results(), f, indent=4)


class ChatSimulation(Simulation):
    '''Responsible for simulating, maintaining, processing states'''

    def __init__(self, max_messages: int = 5, first_turn_role: str = 'user'):
        
        self.max_messages = max_messages
        self.first_turn_role = first_turn_role

        super().__init__()

    async def setup(self, runs: int = 3):
        # simulation components
        self.app: ChatAgent = ChatAgent()
        self.synthetic_users: list[SyntheticUserAgent] = await SyntheticUserFactory() \
                        .as_a('someone who wants help') \
                        .they_want_to('ask a question') \
                        .with_app_objective('answer a question') \
                        .abuild_many(runs)
    
    async def coroutine(self, run_id: int) -> State:
        '''Simulates a chat conversation between the app and a synthetic user agent'''
        
        if not isinstance(self.app, ChatAgent) or not isinstance(self.synthetic_users, list) or not all(isinstance(user, SyntheticUserAgent) for user in self.synthetic_users):
            raise ValueError("App must be of type ChatAgent. Synthetic users must be of type list[SyntheticUserAgent]")
        
        synth_user_agent = self.synthetic_users[run_id]
        print('synth user agent', synth_user_agent)

        print_run_id(run_id)
        print(system_prompt_str(self.app.model_settings))
        print(system_prompt_str(synth_user_agent.model_settings))
        
        max_messages = self.max_messages or 3
        
        state = State()
        if self.first_turn_role == 'user' and max_messages > 0:
            state = await synth_user_agent.aprocess_turn(state)
            print_run_id(run_id)
            print(str_msgs([state.messages[-1]]))

        while True:
            # app turn
            if len(state.messages) > max_messages: break
            state = await self.app.aprocess_turn(state)
            print_run_id(run_id)
            print(str_msgs([state.messages[-1]]))

            # synthetic user turn
            if len(state.messages) > max_messages: break
            state = await synth_user_agent.aprocess_turn(state)
            print_run_id(run_id)
            print(str_msgs([state.messages[-1]]))

        return state

    def export_run_data(self) -> dict:
        return_obj = {'inputs': [], 'outputs': [], 'ground_truths': [], 'metadata': []}        
        for run_id, run_data in self.run_data.items():
            state: State | None = run_data.final_state
            if state:
                return_obj['outputs'].append({'messages': [m.__dict__() for m in state.messages]})
                return_obj['metadata'].append({'run_id': run_id, 'run_data_hash': run_data.compute_hash()})
        return return_obj
