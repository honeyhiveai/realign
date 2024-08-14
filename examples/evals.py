from realign.eval_utils import Eval
from realign.llm_utils import allm_messages_call
from realign.types import evaluator

import asyncio

nums = [1, 2, 3, 4, 10] # mean is 4


@evaluator
def composite(nums):
    r1 = Eval('fourtytwo').call(nums)
    r2 = Eval('pymean').call(nums, weight=20)
    
    return [r1, r2], {'prev_results': [r1, r2]}

@evaluator
async def custom_eval(nums):
    return sum(nums)/len(nums)

@evaluator
def insert_noise(x):
    import random
    return x + random.uniform(-2, 2)

print(Eval(composite).call(nums).str)

r1 = asyncio.run(
    Eval(custom_eval, inject_globals={'insert_noise': insert_noise}).acall(nums, repeat=3)
)
print(r1.str)
