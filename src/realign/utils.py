import asyncio
from typing import Callable, Any, Coroutine

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

async def arun_callables(funcs: list[Callable], args: list[list] = [[]], kwargs: list[dict] = [{}]) -> list[Any]:
    
    if len(funcs) != len(args) or len(funcs) != len(kwargs):
        raise Exception("funcs, args, and kwargs must all be the same length.")

    async def run_func(func, arg, kwarg):
        if asyncio.iscoroutinefunction(func):
            return await func(*arg, **kwarg)
        else:
            return await asyncio.to_thread(func, *arg, **kwarg)

    tasks = [run_func(func, arg, kwarg) for func, arg, kwarg in zip(funcs, args, kwargs)]
    
    results = await asyncio.gather(*tasks)
    return results



def run_async(tasks: list[Coroutine] | Coroutine):
    
    # run the tasks in parallel
    async def gather_async(*tasks):
        return await asyncio.gather(*tasks)
    
    async def single_task(task):
        return await task
    
    if not isinstance(tasks, (list, tuple)):
        coroutine = single_task(tasks)
    else:
        coroutine = gather_async(*tasks)
    
    return asyncio.run(coroutine)

