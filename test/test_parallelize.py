import asyncio

from realign.llm_utils import parallelize

# test parallelize
async def test_parallelize():
    
    # sample async function
    async def add_one(x: int, y: int = 1) -> int:
        return x + y
    
    # sample sync function
    def add_two(x: int, y: int = 2) -> int:
        return x + y
    
    funcs = [add_one, add_two, add_one]
    args = [[1], [2], [3]]
    kwargs = [{"y": 1}, {"y": 2}, {"y": 3}]
    results = await parallelize(funcs, args, kwargs)
    print(results)

asyncio.run(test_parallelize())