import asyncio
import json
import os
import inspect
from typing import Any, Callable, Coroutine, Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv

from realign.datasets import Dataset, ChatDataset
from realign.llm_utils import (
    router, 
    State,
    OpenAIMessage,
    RunData,
    system_prompt_str, 
    str_msgs, 
    print_run_id
)
from realign.agents import (
    AbstractAgent, 
    AgentBuilder, 
    SyntheticUserFactory, 
    SyntheticUserAgent, 
    ChatAgent
)
from realign.evaluators import evaluator, EvalResult
from realign.utils import arun_callables, bcolors


@dataclass
class Context:
    run_id: int
    
    sim_args: list[Any] = field(default_factory=list)
    sim_kwargs: dict[str, Any] = field(default_factory=dict)
    
    initial_state: Optional[State] = None
    final_state: Optional[State] = None
    
    run_data: Optional[RunData] = None
    eval_results: list[EvalResult] = field(default_factory=list)

    def __getitem__(self, key: str | Any):
        if not isinstance(key, str):
            raise TypeError(f"Context keys must be strings, not {type(key)}")
        if not key.isidentifier():
            raise ValueError(f"Context key {key} is not a valid variable name")
        if not hasattr(self, key):
            raise KeyError(f"Context does not have attribute {key}")
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        if not isinstance(key, str):
            raise TypeError(f"Context keys must be strings, not {type(key)}")
        if not key.isidentifier():
            raise ValueError(f"Context key {key} is not a valid variable name")
        setattr(self, key, value)

class Simulation:
    
    def __init__(self, *args, **kwargs):
        
        self.sim_args = args
        self.sim_kwargs = kwargs
        
        self.runs = None
        
        # if provided, set the router settings to honor rate limits
        self.router_settings = None
        
        # evaluators
        self.evaluators = []

        # results
        self.run_data: dict[int, RunData] = dict()
        self.eval_results: dict[int, list[EvalResult]] = dict()

    async def setup(self, *args, **kwargs):
        '''Sets up objects used in the simulation'''
        
        # simulation components accessible to main
        self.dataset: Dataset = None
        self.app: AbstractAgent = None
        self.evaluators: list[Callable | evaluator] = []
        
    async def before_each(self, run_context: Context):
        '''Runs asyncronously before each simulation run'''
        return None
    
    async def main(self, run_context: Context) -> RunData:
        print('Running empty main! Please override.')
    
    async def after_each(self, run_context: Context):
        '''Runs synchronously after each simulation run'''
        
        # run the evaluators
        if self.evaluators and len(self.evaluators) > 0:
            
            # NOTE: final_state is assumed to be read only
            args = [(run_context.final_state,) for _ in range(len(self.evaluators))]
            
            eval_scores = await arun_callables(funcs=self.evaluators,
                                                args=args)
            
            # save the evaluation results
            run_context.eval_results = eval_scores
            
            for e in range(len(self.evaluators)):
                if isinstance(self.evaluators[e], evaluator) and isinstance(self.evaluators[e].prev_run, EvalResult):
                    # results
                    run_context.eval_results[e] = self.evaluators[e].prev_run.copy()
                else:
                    # scores
                    run_context.eval_results[e] = eval_scores[e]
            
        else:
            run_context.eval_results = []
    
        
    async def windup(self):
        '''Runs synchronously after all simulation runs'''
        
        # aggregate the results
        for run_id in range(self.runs):
            self.run_data[run_id] = self.run_contexts[run_id].run_data
            self.eval_results[run_id] = self.run_contexts[run_id].eval_results
            
        self.push_runs_to_dataset()
        self.push_evals_dataset()
        
        # print the results for each run
        for run_id in range(self.runs):
            print_run_id(run_id)
            print(self.run_data[run_id].final_state, '\n\n')
    
    async def run_concurrently(self, run_context: Context):
        
        # before_each
        await self.before_each(run_context)
            
        # run the simulation main
        run_context.final_state = await self.main(run_context)
        run_context.run_data = RunData(run_context.final_state,
                                          run_id=run_context.run_id)
             
        # after_each
        await self.after_each(run_context)
     
    async def run_simulation(self):
        
        # start timer
        start = asyncio.get_event_loop().time()
        
        try:
            # set up the thread contexts
            self.run_contexts: list[Context] = [
                Context(run_id, self.sim_args, self.sim_kwargs) 
                for run_id in range(self.runs)
            ]
            
            # setup
            await self.setup(*self.sim_args, **self.sim_kwargs)
            
            # before_each, main, after_each
            simulation_run_tasks = [
                self.run_concurrently(self.run_contexts[run_id])
                for run_id in range(self.runs)
            ]
            await asyncio.gather(*simulation_run_tasks)
            
            # windup
            self.final_states = [
                run_context.final_state
                for run_context in self.run_contexts
            ]
            await self.windup()

        finally:
            if router:
                await router.shutdown()
                router_stats = router.get_stats()
            else:
                router_stats = None

            # end timer
            end = asyncio.get_event_loop().time()
            
            if self.evaluators and len(self.evaluators) > 0:
                for run_id in range(self.runs):
                    print_run_id(run_id)
                    self.print_evals(self.run_contexts[run_id])

            print('\n\n' + '-'*100)
            print('Simulation Stats:')
            print(f'\tSimulation Duration: {(end - start):.3f} sec')
            print(f'\tRuns: {self.runs} runs')
            print('-'*100)
            
            for model_name, model_stats in router_stats.items():
                print(f'{model_name} Stats:')
                print(json.dumps(model_stats, indent=4).replace('"', ''))
            print('-'*100 + '\n\n')

    def run(self, runs: int = 3) -> 'Simulation':
        '''Runs the main event loop for the simulation'''

        # simulation is fundamentally a main that runs N times
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
    
    def print_evals(self, run_context: Context):
        print(bcolors.WARNING)
        for i, e in enumerate(run_context.eval_results):
            print(self.evaluators[i].name, ":", e)
            print("- " * 50)
        print(bcolors.ENDC)    
    
    def export_run_data(self) -> dict:
        return_obj = {'inputs': [], 'outputs': [], 'ground_truths': [], 'metadata': []}    
        for run_id, run_data in self.run_data.items():
            state: State | None = run_data.final_state
            if state:
                return_obj['outputs'].append({'state': str(state)})
                return_obj['metadata'].append({'run_id': run_id, 'run_data_hash': run_data.compute_hash()})
        return return_obj
    
    def export_eval_results(self) -> dict:
        data_obj = {'run_data_hash': [], 'metadata': [], 'evaluations': []}
        for run_id, eval_results in self.eval_results.items():
            data_obj['run_data_hash'].append(self.run_data[run_id].compute_hash())
            eval_dict = dict()
            for i, e in enumerate(eval_results):
                if isinstance(e, EvalResult):
                    eval_dict[e.func_impl] = e.str
                else:
                    eval_dict[i] = e
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
    def __init__(self, *args, max_messages: int = 5, first_turn_role: str = 'user', **kwargs):
        
        self.max_messages = max_messages
        self.first_turn_role = first_turn_role

        super().__init__(*args, **kwargs)
    
    def export_run_data(self) -> dict:
        return_obj = {'inputs': [], 'outputs': [], 'ground_truths': [], 'metadata': []}        
        for run_id, run_data in self.run_data.items():
            state: State | list | None = run_data.final_state
            if state:
                if isinstance(state, State):
                    return_obj['outputs'].append({'messages': [m.__dict__() for m in state.messages]})
                elif isinstance(state, list) and all(isinstance(s, OpenAIMessage) for s in state):
                    return_obj['outputs'].append({'messages': [m.__dict__() for m in state]})
                else:
                    return_obj['outputs'].append({'messages': '<invalid output>'})
                
                return_obj['metadata'].append({'run_id': run_id, 'run_data_hash': run_data.compute_hash()})
        return return_obj


class OldChatSimulation(Simulation):
    '''Responsible for simulating, maintaining, processing states'''

    def __init__(self, *args, max_messages: int = 5, first_turn_role: str = 'user', **kwargs):
        
        self.max_messages = max_messages
        self.first_turn_role = first_turn_role

        super().__init__(*args, **kwargs)

    async def setup(self, *args, **kwargs):
        await super().setup()
        
        # simulation components
        self.app: ChatAgent = ChatAgent()
        self.synthetic_users: list[SyntheticUserAgent] = await SyntheticUserFactory() \
                        .as_a('someone who wants help') \
                        .they_want_to('ask a question') \
                        .with_app_objective('answer a question') \
                        .abuild_many(self.runs)
    
    async def main(self, run_context: Context) -> State:
        '''Simulates a chat conversation between the app and a synthetic user agent'''
        
        if not isinstance(self.app, ChatAgent) or not isinstance(self.synthetic_users, list) or not all(isinstance(user, SyntheticUserAgent) for user in self.synthetic_users):
            raise ValueError("App must be of type ChatAgent. Synthetic users must be of type list[SyntheticUserAgent]")
        
        synth_user_agent = self.synthetic_users[run_context.run_id]
        print('synth user agent', synth_user_agent)

        print_run_id(run_context.run_id)
        print(system_prompt_str(self.app.agent_settings))
        print(system_prompt_str(synth_user_agent.agent_settings))
        
        max_messages = self.max_messages or 3
        
        state = State()
        if self.first_turn_role == 'user' and max_messages > 0:
            state = await synth_user_agent.aprocess_turn(state)
            print_run_id(run_context.run_id)
            print(str_msgs([state.messages[-1]]))

        while True:
            # app turn
            if len(state.messages) > max_messages: break
            state = await self.app.aprocess_turn(state)
            print_run_id(run_context.run_id)
            print(str_msgs([state.messages[-1]]))

            # synthetic user turn
            if len(state.messages) > max_messages: break
            state = await synth_user_agent.aprocess_turn(state)
            print_run_id(run_context.run_id)
            print(str_msgs([state.messages[-1]]))

        # return the final state
        return state

    def export_run_data(self) -> dict:
        return_obj = {'inputs': [], 'outputs': [], 'ground_truths': [], 'metadata': []}        
        for run_id, run_data in self.run_data.items():
            state: State | list | None = run_data.final_state
            if state:
                if isinstance(state, State):
                    return_obj['outputs'].append({'messages': [m.__dict__() for m in state.messages]})
                elif isinstance(state, list) and all(isinstance(s, OpenAIMessage) for s in state):
                    return_obj['outputs'].append({'messages': [m.__dict__() for m in state]})
                else:
                    return_obj['outputs'].append({'messages': '<invalid output>'})
                
                return_obj['metadata'].append({'run_id': run_id, 'run_data_hash': run_data.compute_hash()})
        return return_obj
