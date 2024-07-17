import asyncio
from functools import wraps
from realign.types import RunData
from typing import Any

# this object contains a single score
# the score can be of any type, including a list of scores
class Evaluation:
    def __init__(self, score: Any, result: bool | None, explanation: str | None = None, run_data: RunData = None, eval_name: str | None = None, repeat: int =1):
        self.score = score
        self.result = result
        self.explanation = explanation
        self.run_data = run_data
        self.eval_name = eval_name
        self.repeat = repeat

    def __repr__(self):
        # get the object id of the run_data
        run_data_id = id(self.run_data) if self.run_data else None
        return f'(eval_name: {self.eval_name}, run_data: {run_data_id}, score: {self.score}, result: {self.result})'
    
    def __str__(self):
        return self.__repr__()
    
    def unpack(self):
        if self.explanation:
            return self.score, self.result, self.explanation
        return self.score, self.result
    
    def __dict__(self):
        return {
            self.eval_name: {
                'score': self.score,
                'result': self.result,
                'explanation': self.explanation,
            }
        }
    
    def to_dict(self):
        return self.__dict__()

# evaluator decorator to wrap an app scorer inside a Evaluation object
# TODO: composable evaluators
def evaluator(eval_func=None, *, repeat=1) -> Evaluation:
    def decorator(func):
        @wraps(eval_func)
        async def wrapper(run_data: RunData, *args, **kwargs):
            async def single_run():
                if asyncio.iscoroutinefunction(func):
                    # If the eval_func is already a coroutine function, just await it
                    response = await func(run_data.final_state, 
                                          *args,
                                          **kwargs
                                          )
                else:
                    # If it's a regular function, run it in a thread pool
                    response = await asyncio.to_thread(func, 
                                                       run_data.final_state,
                                                       *args,
                                                       **kwargs
                                                        )

                # verify that the eval_func response is a tuple of 3 elements (score: Any, result: bool | None, explanation: str | None)
                if not 2 <= len(response) <= 3 or \
                    type(response) not in [list, tuple] or \
                        type(response[1]) not in [bool, type(None)] or \
                            (len(response) == 3 and type(response[2]) not in [str, type(None)]):
                    raise ValueError('Evaluator response must be a tuple of 2 elements, the score of type Any and result of type bool | None')

                # unpack results
                score = None
                result = None
                explanation = None
                if len(response) == 3:
                    score, result, explanation = response
                    return (score, result, explanation)

                score, result = response
                return (score, result, None)
            
            assert repeat >= 0, 'Repeat must be greater than 0'
            
            if repeat == 0:
                return None

            if repeat > 1:
                tasks = [single_run() for _ in range(repeat)]
                score_result_tuples = await asyncio.gather(*tasks)
                scores, results, explanations = zip(*score_result_tuples)

                # for repeats, return the full array of scores and results
                return Evaluation(
                    scores,
                    results,
                    explanations,
                    run_data,
                    func.__name__,
                    repeat
                )
            else:
                # for single runs, return a single score and result
                score, result, explanation = await single_run()
                return Evaluation(
                    score,
                    result,
                    explanation,
                    run_data,
                    func.__name__,
                    repeat
                )

        return wrapper

    if eval_func is None:
        return decorator
    else:
        return decorator(eval_func)
