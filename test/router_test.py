from realign.router import Router

import asyncio
import time

router = Router()


async def api_call():
    
    response = await router.acompletion(model='openai/gpt-4o-mini', 
                             messages=[{'role': 'user', 'content': 'What is the capital of France in 1 word?'}])
    
    return response.choices[0].message.content

async def main():
    for _ in range(5):
        tasks = [api_call() for _ in range(20)]
        responses: list[str] = await asyncio.gather(*tasks)

        await asyncio.sleep(2)

        for r in responses:
            assert 'paris' in r.lower()

def run():
    
    # get a new event loop
    loop = asyncio.new_event_loop()
    
    # set the event loop
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main())
    finally:
        # Shut down the router
        loop.run_until_complete(router.shutdown())

        # Close the loop
        loop.close()
        
def run():
    # start the timer
    start = time.time()
    try:
        # Use asyncio.run which handles loop creation and cleanup
        asyncio.run(run_main())
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Process finished in ")

async def run_main():

    try:
        await main()
    finally:
        if router:
            await router.shutdown()

run()
