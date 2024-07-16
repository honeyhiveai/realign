import asyncio
from functools import wraps
from realign.types import RunData

# this object contains a single score
# the score can be of any type, including a list of scores
class Evaluation(object):
    def __init__(self, score, result, run_data=None, eval_name=None, repeat=1):
        self.score = score
        self.result = result
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
        return self.score, self.result
    
    def __dict__(self):
        return {
            'eval_name': self.eval_name,
            'score': self.score,
            'result': self.result
        }
    
    def to_dict(self):
        return self.__dict__()

# evaluator decorator to wrap an app scorer inside a Evaluation object
def evaluator(eval_func=None, *, repeat=1):
    def decorator(func):
        @wraps(eval_func)
        async def wrapper(run_data: RunData, *args, **kwargs):
            async def single_run():
                if asyncio.iscoroutinefunction(func):
                    # If the eval_func is already a coroutine function, just await it
                    response = await func(run_data.final_state, *args, **kwargs)
                else:
                    # If it's a regular function, run it in a thread pool
                    response = await asyncio.to_thread(func, run_data.final_state, *args, **kwargs)

                # verify that the eval_func response is a tuple of 2 elements (score: Any, result: bool | None)
                if type(response) not in [list, tuple] or \
                    len(response) != 2 or \
                        type(response[1]) not in [bool, type(None)]:
                    raise ValueError('Evaluator response must be a tuple of 2 elements, the score of type Any and result of type bool | None')
                    
                # unpack results
                score, result = response
                return (score, result)

            if repeat > 1:
                tasks = [single_run() for _ in range(repeat)]
                score_result_tuples = await asyncio.gather(*tasks)
                scores, results = zip(*score_result_tuples)

                # we return the full array of scores and whether all results passed
                return Evaluation(scores, all(results), run_data, func.__name__, repeat)
            else:
                score, result = await single_run()
                return Evaluation(score, result, run_data, func.__name__)

        return wrapper

    if eval_func is None:
        return decorator
    else:
        return decorator(eval_func)
