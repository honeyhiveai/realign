import asyncio
from typing import Callable, Any

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
    """
    Runs a list of functions (sync or async) in parallel using asyncio.gather.

    Args:
        funcs (list[Callable]): The list of functions to run
        args (list[list]): The list of arguments to pass to each function
        kwargs (list[dict]): The list of keyword arguments to pass to each function

    Returns:
        list[Any]: A list of the results of each function
    """
    
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