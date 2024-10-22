from state import GlobalState
from event import Idea
from controller import Controller
from process import Process
from tsafepq import ThreadSafePriorityQueue

from datasets import load_dataset
import threading
import curses
import time
import asyncio
from itertools import cycle
from typing import Optional
import os
import sys
import contextlib
import io

class bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def draw_queue_contents(stdscr, queue_names: list[str]):
    stdscr.clear()
    
    try:
        height, width = stdscr.getmaxyx()
        
        for i, queue_name in enumerate(queue_names):
            queue: ThreadSafePriorityQueue = GlobalState.get_queue(queue_name)
            y_pos = i * (height // len(queue_names))
            
            stdscr.addstr(y_pos, 0, f"{queue_name}:", curses.color_pair(i + 1))
            stdscr.addstr(y_pos + 1, 2, f"Size: {queue.size()}", curses.color_pair(i + 1))
            
            items: list[Idea] = queue.peek_many(10)
            for j, item in enumerate(items):  # Show top 10 items
                if y_pos + j + 2 < height:
                    stdscr.addstr(y_pos + j + 2, 2, f"{item.abs_rating}/5: {item.seed}", curses.color_pair(i + 1))

            if queue.size() > 10:
                stdscr.addstr(y_pos + 12, 2, f"... and {queue.size() - 10} more", curses.color_pair(i + 1))
    except curses.error:
        pass                
    
    stdscr.refresh()

def visualize_queues():
    def run_visualization(stdscr):
        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)

        curses.curs_set(0)  # Hide the cursor
        queue_names = ['seed_queue', 'explore_queue', 'staging_queue', 'analysis_queue']
        
        while not GlobalState.stop_event.is_set():
            draw_queue_contents(stdscr, queue_names)
            
            time.sleep(0.3)
    
    curses.wrapper(run_visualization)


# Simultaneous Research And Content System (CRACS)

class Main:
    
    FREQ_MULTIPLIER = 2
    
    def __init__(self):
        
        # Load and shuffle the dataset
        persona_hub = load_dataset("proj-persona/PersonaHub", "persona")["train"]
        shuffled_personas = persona_hub.shuffle()
        persona_cycle = cycle(shuffled_personas)
        
        # create the controller
        self.controller: Controller = Controller()
        
        self.vis_thread: threading.Thread = None

        # SEED GENERATOR
        async def seed_generator(self: Process, persona_cycle: cycle):
            print(self.name, "started")
            seed_queue: ThreadSafePriorityQueue = GlobalState.get_queue('seed_queue')
            BATCH_SIZE = 5
            while not GlobalState.stop_event.is_set():
                if seed_queue.size() < 10:
                    persona_batch = [next(persona_cycle)['persona'] for _ in range(BATCH_SIZE)]
                    idea: Idea = Idea(
                        seed='Group: ' + ', '.join(persona_batch),
                        depth=0,
                        lineage=[]
                    )
                    seed_queue.push(idea)
                    print(f"{self.name}: Pushed idea {idea.seed} to seed_queue")
                    await asyncio.sleep(1/(self.polling_freq * Main.FREQ_MULTIPLIER))
            print(self.name, "stopped")
        
        # IDEA PUSHER RATER
        async def idea_pusher_rater(self: Process):
            print(self.name, "started")
            BATCH_SIZE = 3
            seed_queue: ThreadSafePriorityQueue = GlobalState.get_queue('seed_queue')
            explore_queue: ThreadSafePriorityQueue = GlobalState.get_queue('explore_queue')
            while not GlobalState.stop_event.is_set():
                ideas_to_rate = []
                for _ in range(BATCH_SIZE):
                    idea: Optional[Idea] = seed_queue.poll()
                    if idea:
                        print(f"{self.name}: Polled idea {idea.seed} from seed_queue")
                        ideas_to_rate.append(idea)
                if ideas_to_rate:
                    rated_ideas = await asyncio.gather(
                        *[GlobalState.rate_absolute(self.client, idea) for idea in ideas_to_rate]
                    )
                    for idea, rated_idea in zip(ideas_to_rate, rated_ideas):
                        idea.abs_rating = rated_idea
                        explore_queue.push(idea)
                        print(f"{self.name}: Pushed idea {idea.seed} to explore_queue")
                else:
                    print(f"{self.name}: Seed_queue is empty")
                await asyncio.sleep(1/(self.polling_freq * Main.FREQ_MULTIPLIER))
        
        # IDEA EVOLVER
        async def idea_evolver(self: Process):
            print(self.name, "started")
            BATCH_SIZE = 5
            explore_queue: ThreadSafePriorityQueue = GlobalState.get_queue('explore_queue')
            staging_queue: ThreadSafePriorityQueue = GlobalState.get_queue('staging_queue')
            while not GlobalState.stop_event.is_set():
                ideas_to_evolve = []
                for _ in range(BATCH_SIZE):
                    idea: Optional[Idea] = explore_queue.poll()
                    if idea:
                        ideas_to_evolve.append(idea)
                evolved_ideas = await asyncio.gather(
                    *[GlobalState.evolve_idea(self.client, idea) for idea in ideas_to_evolve]
                )
                for new_idea in evolved_ideas:
                    if new_idea.depth < 1:
                        explore_queue.push(new_idea) # push to explore_queue
                    else:
                        staging_queue.push(new_idea) # push to staging_queue
                        print(f"{self.name}: Pushed idea {new_idea.seed} to staging_queue")
                        
                await asyncio.sleep(1/(self.polling_freq * Main.FREQ_MULTIPLIER))
                
                
        # IDEA ANALYZER
        async def idea_analyzer(self: Process):
            print(self.name, "started")
            BATCH_SIZE = 1
            staging_queue: ThreadSafePriorityQueue = GlobalState.get_queue('staging_queue')
            analysis_queue: ThreadSafePriorityQueue = GlobalState.get_queue('analysis_queue')
            while not GlobalState.stop_event.is_set():
                ideas_to_analyze = []
                for _ in range(BATCH_SIZE):
                    idea: Optional[Idea] = staging_queue.poll()
                    if idea:
                        print(f"{self.name}: Polled idea {idea.seed} from staging_queue")
                        ideas_to_analyze.append(idea)
                if len(ideas_to_analyze) > 0:
                    analyzed_ideas = await asyncio.gather(
                        *[GlobalState.analyze_idea(self.client, idea) for idea in ideas_to_analyze]
                    )
                    for analyzed_idea in analyzed_ideas:
                        analysis_queue.push(analyzed_idea)
                        print(f"{self.name}: Pushed analyzed idea {analyzed_idea.seed} to analysis_queue")
                else:
                    print(f"{self.name}: Staging_queue is empty")
                await asyncio.sleep(1/(self.polling_freq * Main.FREQ_MULTIPLIER))


        
        GlobalState.add_process(
            Process(
                name='seed_generator',
                process=seed_generator,
                params=(persona_cycle,),
                polling_freq=10
            )
        )
        GlobalState.add_process(
            Process(
                name='idea_pusher_rater',
                process=idea_pusher_rater,
                params=(),
                polling_freq=5
            )
        )    
        GlobalState.add_process(
            Process(
                name='idea_evolver',
                process=idea_evolver,
                params=(),
                polling_freq=2
            )
        )
        
        GlobalState.add_process(
            Process(
                name='idea_analyzer',
                process=idea_analyzer,
                params=(),
                polling_freq=1
            )
        )
    
        # GlobalState.add_queue('broadcast_queue', ThreadSafePriorityQueue(GlobalState))
        GlobalState.add_queue('seed_queue', ThreadSafePriorityQueue(heuristic_func=lambda x: 1))
        GlobalState.add_queue('explore_queue', ThreadSafePriorityQueue(heuristic_func=lambda x: x.abs_rating))
        GlobalState.add_queue('staging_queue', ThreadSafePriorityQueue(heuristic_func=lambda x: x.depth * x.abs_rating, stop_size=3))
        GlobalState.add_queue('analysis_queue', ThreadSafePriorityQueue(heuristic_func=lambda x: x.abs_rating))
    
    def stop(self):
        
        if self.vis_thread:
            self.vis_thread.join()
        
        # Stop the Controller
        self.controller.stop_processes()
    
    def start(self, runtime_seconds: Optional[int] = None):
        
        # Start the visualization in a separate thread
        if not GlobalState.debug:
            self.vis_thread: threading.Thread = threading.Thread(
                target=visualize_queues
            )       
            self.vis_thread.start()

        # Start the Controller
        self.controller.start_processes(runtime_seconds=runtime_seconds)


if __name__ == '__main__':
    messages = []
    
    GlobalState.debug = False
    
    while True:
        GlobalState.reset()
        main = Main()
        
        user_input = input(f"\n{bcolor.OKCYAN}Enter a prompt (or 'q' to exit): {bcolor.ENDC}")
        if user_input.lower() in ['quit', 'q']:
            break
        
        messages.append({"role": "user", "content": "Incorporate this feedback into your brainstorming process: " + user_input})
        GlobalState.messages = messages.copy()
        
        # prevent stdout from being captured
        with contextlib.redirect_stdout(io.StringIO()) if not GlobalState.debug else contextlib.nullcontext():
            try:
                main.start()
            except KeyboardInterrupt:
                print("Controller: KeyboardInterrupt received.")
            finally:
                time.sleep(5)
                main.stop()
            
        final_ideas = GlobalState.get_queue('analysis_queue').peek_many(5)
        print(f"{bcolor.OKGREEN}Top 5 ideas:")
        for i, idea in enumerate(final_ideas):
            print(f" {i+1}. {idea.abs_rating}/5: {idea.seed}\n")
        print(f"--------------------------------{bcolor.ENDC}\n")
        
        # messages.append({"role": "assistant", "content": '\n'.join([idea.seed for idea in final_ideas])})
        GlobalState.messages = messages.copy()
        

    print("Program terminated. :)")
