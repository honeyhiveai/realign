import asyncio
from typing import Callable, Any, Coroutine, Optional

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
    
async def arun_callables(funcs: list[Callable], 
                         args: Optional[list[list]] = None, 
                         kwargs: Optional[list[dict]] = None) -> list[Any]:
    if args is None:
        args = [[]] * len(funcs)
    if kwargs is None:
        kwargs = [{}] * len(funcs)

    if len(funcs) != len(args) or len(funcs) != len(kwargs):
        raise ValueError("funcs, args, and kwargs must all be the same length.")

    async def run_task(index: int, 
                       func: Callable, 
                       arg: list, 
                       kwarg: dict) -> tuple[int, Any]:
        
        is_coro = (hasattr(func, '__call__') and asyncio.iscoroutinefunction(func.__call__)) or asyncio.iscoroutinefunction(func)
        
        if is_coro:
            result = await func(*arg, **kwarg)
        else:
            result = await asyncio.to_thread(func, *arg, **kwarg)
        return index, result

    tasks = [
        asyncio.create_task(run_task(i, func, arg, kwarg))
        for i, (func, arg, kwarg) in enumerate(zip(funcs, args, kwargs))
    ]

    results = await asyncio.gather(*tasks)
    return [result for _, result in sorted(results, key=lambda x: x[0])]



def run_async(tasks: list[Coroutine] | Coroutine) -> Any | Coroutine:
    
    # run the tasks in parallel
    async def gather_async(*tasks):
        return await asyncio.gather(*tasks)
    
    async def single_task(task):
        return await task
    
    if not isinstance(tasks, (list, tuple)):
        coroutine = single_task(tasks)
    else:
        coroutine = gather_async(*tasks)
        
    # if we are in a running event loop, return coroutine
    if asyncio.get_event_loop().is_running():
        return coroutine
    
    return asyncio.run(coroutine)

