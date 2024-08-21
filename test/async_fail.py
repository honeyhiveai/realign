import asyncio

def sync_function():
    loop = asyncio.get_running_loop()
    result = loop.run_until_complete(async_task())
    return result

async def async_task():
    await asyncio.sleep(1)
    return "Task completed"

async def main():
    print("Starting main")
    result = sync_function()
    print(result)
    print("Main completed")

if __name__ == "__main__":
    asyncio.run(main())