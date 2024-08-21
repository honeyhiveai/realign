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

async def arun_callables(funcs: list[Callable], args: list[list] = [[]], kwargs: list[dict] = None) -> list[Any]:
    
    if len(funcs) != len(args):
        raise Exception("funcs and args must be the same length.")
    
    if kwargs:
        if len(funcs) != len(kwargs):
            raise Exception("funcs, args, and kwargs must all be the same length.")
    else:
        # empty kwargs
        kwargs = [dict() for _ in range(len(funcs))]

    async_tasks = []
    sync_tasks = []
    for func, arg, kwarg in zip(funcs, args, kwargs):
        if asyncio.iscoroutinefunction(func):
            async_tasks.append(func(*arg, **kwarg))
        else:
            sync_tasks.append((func, arg, kwarg))

    # run the async tasks in parallel
    async_results = await asyncio.gather(*async_tasks)
    
    # run the sync tasks in serial
    sync_results = []
    for func, arg, kwarg in sync_tasks:
        sync_results.append(func(*arg, **kwarg))
    # combine the async and sync results
    results = async_results + sync_results
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

