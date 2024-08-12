import asyncio
import time
from concurrent.futures import ProcessPoolExecutor

def cpu_bound_worker(x, y):
    print("In worker")
    time.sleep(3)  # Simulating CPU-bound work
    return x + y

async def io_bound_coroutine():
    await asyncio.sleep(1)  # Simulating I/O-bound work
    print("Done with I/O-bound coroutine")

async def main():
    print("Starting main coroutine")
    
    # Schedule the I/O-bound coroutine
    io_task = asyncio.create_task(io_bound_coroutine())
    
    # Run the CPU-bound function in the process pool
    loop = asyncio.get_running_loop()
    cpu_task = loop.run_in_executor(None, cpu_bound_worker, 3, 4)
    
    # Wait for both tasks to complete
    done, pending = await asyncio.wait([io_task, cpu_task])
    
    # Get the result of the CPU-bound task
    for task in done:
        if task == cpu_task:
            result = task.result()
            print(f"CPU-bound task result: {result}")
    
    print("Main coroutine finished")

if __name__ == "__main__":
    asyncio.run(main())