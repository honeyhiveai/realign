import os
import sys
from .configs import config
from .evaluators import evaluator, aevaluator
from .llm_utils import allm_messages_call, llm_messages_call, run_async
from .simulation import Simulation, ChatSimulation, Context

try:
    # Get the directory containing the file being run
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, use the sys._MEIPASS
        running_dir = sys._MEIPASS
    else:
        # If it's not bundled, use the directory containing the script
        running_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    # Prepend the directory containing the running file to config.path
    config_path = os.path.join(running_dir, 'config.yaml')
    if os.path.exists(config_path):
        config.path = config_path
    else:
        print(f'Warning: no config file found. Please set it using\nimport realign\nrealign.config.path = "path/to/config.yaml"')

except Exception as e:
    print(f'Error while loading config file: {e}')
    print('Continuing without config file...')

__all__ = [
    'configs',
    'config',
    'load_config',
    'evaluator',
    'aevaluator',
    'llm_messages_call',
    'allm_messages_call',
    'run_async',
    'Simulation',
    'ChatSimulation',
    'Context',
]