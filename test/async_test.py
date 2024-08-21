import asyncio

async def api_call(x): return x

async def build():
    tasks = [asyncio.create_task(api_call('london')) for _ in range(5)]
    responses: list[str] = await asyncio.gather(*tasks)
    return responses

async def abuild_many():
    return await asyncio.gather(*[build() for _ in range(5)])
    
def build_many():
    return asyncio.run(abuild_many())

def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    print(build_many())
    
    tasks = [api_call('paris') for _ in range(20)]
    responses = loop.run_until_complete(asyncio.gather(*tasks))

    loop.close()
    
    print(responses)

main()


async def slow_task(i, delay):
    await asyncio.sleep(delay)
    print(f"Task {i} done")
    return delay

async def main():
    delays = [3, 2, 1]
    tasks = [asyncio.create_task(slow_task(i, delays[i])) for i in range(3)]


# if __name__ == '__main__':
#     asyncio.run(main())
