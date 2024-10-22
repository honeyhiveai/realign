import threading
from typing import Coroutine
import asyncio
import openai

from state import GlobalState

class Process(threading.Thread):
    """
    Represents a processing thread that consumes ideas from an input queue,
    processes them, and optionally adds new ideas to output queues.
    """
    
    def __init__(
        self, 
        name: str, 
        process: Coroutine,
        params: tuple = (),
        polling_freq: float = 0.5,
    ):
        """
        Initializes the Process.
        
        Args:
            name (str): Name of the process for identification.
            global_state (GlobalState): The shared global state.
            input_queue (ThreadSafePriorityQueue): The queue from which to consume ideas.
            output_queues (list[ThreadSafePriorityQueue]): The queues to which to push processed ideas.
            process (Coroutine): The coroutine to run for processing ideas.
        """
        super().__init__(name=name)
        self.process = process
        self.params = params
        self.polling_freq = polling_freq
        
        self.client = openai.AsyncOpenAI()
        
    def run(self):
        """
        The main loop of the process.
        """
        try:
            print(f"Process '{self.name}': Started.")
            while not GlobalState.stop_event.is_set():
                asyncio.run(self.process(self, *self.params))
            
            print(f"Process '{self.name}': Stopped.")
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        
            


