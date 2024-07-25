import asyncio
import json
from typing import Any, Dict, List

class BaseClass:
    def __init__(self, subroutine: Any = None, runs: int = 1):
        self.subroutine = subroutine
        self.runs = runs

        # simulation/evaluation components
        self.dataset = None
        self.evaluators = []

        # results
        self.run_data: Dict[int, Any] = {}
        self.eval_results: Dict[int, Any] = {}

    async def subroutine_with_evals(self, run_id: int, **subroutine_kwargs) -> Any:
        if not self.subroutine:
            raise ValueError("Subroutine must be defined")

        # run the subroutine
        final_state = await self.subroutine(run_id, **subroutine_kwargs)

        # wrap the run as an object
        run_data = self.create_run_data(final_state, run_id)

        # save the run data
        self.run_data[run_id] = run_data

        # run the evaluators
        eval_tasks = [asyncio.create_task(eval_func(run_data)) for eval_func in self.evaluators]

        # await all the evaluators
        evals: List[Any] = await asyncio.gather(*eval_tasks)

        # save the evaluation results
        self.eval_results[run_id] = evals

        return final_state

    def create_run_data(self, final_state: Any, run_id: int) -> Any:
        raise NotImplementedError("create_run_data must be defined in the subclass")

    def run(self) -> 'BaseClass':
        raise NotImplementedError("run must be defined in the subclass")

    def subroutine(self, run_id: int, **subroutine_kwargs) -> Any:
        # Define the subroutine method to be callable
        pass

    def export_eval_results(self) -> Dict:
        data_obj = {'run_data_hash': [], 'metadata': [], 'evaluations': []}
        for run_id, evals in self.eval_results.items():
            data_obj['run_data_hash'].append(self.run_data[run_id].compute_hash())
            for evaluation_obj in evals:
                data_obj['evaluations'].append(evaluation_obj.to_dict())
        return data_obj

    def push_evals_dataset(self, evaluations_path: str) -> None:
        with open(evaluations_path, 'w') as f:
            json.dump(self.export_eval_results(), f)
