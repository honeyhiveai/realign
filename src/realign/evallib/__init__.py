
from typing import Callable
import inspect
import warnings

import itertools
import math
import random
import string
import re

from realign.evaluators import evaluator, EvalResult, EvalSettings
from realign.utils import bcolors

# ------------------------------------------------------------------------------
# LIBRARY OF EVALUATORS
# ------------------------------------------------------------------------------


def import_module(module_name):
    try:
        module = __import__(module_name, fromlist=['*'])
        # Add all non-private attributes to the global namespace
        globals().update({name: getattr(module, name) for name in dir(module) if not name.startswith('_')})
        return module
    except ImportError as e:
        warnings.warn(f"{bcolors.FAIL}Failed to import {module_name}: {str(e)}. Some functionality may be unavailable.{bcolors.ENDC}")
        return None


# Attempt imports with warnings
hf = import_module('realign.evallib.hf')
checkers = import_module('realign.evallib.checkers')
stats = import_module('realign.evallib.stats')
llm = import_module('realign.evallib.llm')

# ------------------------------------------------------------------------------
# LOAD ALL EVALUATORS 
# ------------------------------------------------------------------------------

EXCLUDE_GLOBALS = [
    'get_evallib_functions',
    'get_realign_evals_utils',
    'get_python_globals',
    'evaluator',
    'load_static_eval_funcs',
]

def get_evallib_functions(eval_type: str | None = None) -> Callable | None:
    global_vars = globals()
    
    # Filter out the get_evallib_functions function itself
    filtered_globals = {k: v for k, v in global_vars.items() 
                        if k not in EXCLUDE_GLOBALS and not k.startswith('__')}
    
    # Further filter to include only callable items (functions, classes)
    callable_globals = {k: v for k, v in filtered_globals.items() 
                        if callable(v) and inspect.isfunction(v)}
    
    if eval_type is None:
        return callable_globals
    
    assert isinstance(eval_type, str), 'eval_type must be a string'
    
    return callable_globals.get(eval_type)

def get_realign_evals_utils():
    return {
        'evaluator': evaluator,
        'EvalSettings': EvalSettings,
        'EvalResult': EvalResult,
    }

def get_python_globals():
    
    return {
        # Numeric functions
        'abs': abs,
        'round': round,
        'pow': pow,
        'sum': sum,
        'min': min,
        'max': max,
        'divmod': divmod,

        # Type conversion
        'int': int,
        'float': float,
        'complex': complex,
        'bool': bool,
        'str': str,
        'list': list,
        'tuple': tuple,
        'set': set,
        'dict': dict,
        'frozenset': frozenset,

        # Sequence/collection functions
        'len': len,
        'sorted': sorted,
        'reversed': reversed,
        'enumerate': enumerate,
        'zip': zip,
        'any': any,
        'all': all,
        'filter': filter,
        'map': map,

        # String functions
        'chr': chr,
        'ord': ord,

        # Introspection
        'isinstance': isinstance,
        'issubclass': issubclass,
        'hasattr': hasattr,
        'getattr': getattr,
        'setattr': setattr,
        'delattr': delattr,
        'callable': callable,

        # Math module functions
        'math.ceil': math.ceil,
        'math.floor': math.floor,
        'math.trunc': math.trunc,
        'math.sqrt': math.sqrt,
        'math.exp': math.exp,
        'math.log': math.log,
        'math.log10': math.log10,
        'math.sin': math.sin,
        'math.cos': math.cos,
        'math.tan': math.tan,

        # Random module functions
        'random.random': random.random,
        'random.randint': random.randint,
        'random.choice': random.choice,
        'random.shuffle': random.shuffle,

        # Iteration tools
        'itertools.chain': itertools.chain,
        'itertools.cycle': itertools.cycle,
        'itertools.repeat': itertools.repeat,
        'itertools.count': itertools.count,

        # File operations
        'open': open,

        # Other useful builtins
        'range': range,
        'slice': slice,
        'iter': iter,
        'next': next,
        'format': format,
        'hash': hash,
        'id': id,
        'type': type,
        
        # String methods
        'str.lower': str.lower,
        'str.upper': str.upper,
        'str.capitalize': str.capitalize,
        'str.title': str.title,
        'str.swapcase': str.swapcase,
        'str.strip': str.strip,
        'str.lstrip': str.lstrip,
        'str.rstrip': str.rstrip,
        'str.split': str.split,
        'str.rsplit': str.rsplit,
        'str.splitlines': str.splitlines,
        'str.join': str.join,
        'str.replace': str.replace,
        'str.startswith': str.startswith,
        'str.endswith': str.endswith,
        'str.find': str.find,
        'str.rfind': str.rfind,
        'str.index': str.index,
        'str.rindex': str.rindex,
        'str.count': str.count,
        'str.isalnum': str.isalnum,
        'str.isalpha': str.isalpha,
        'str.isdigit': str.isdigit,
        'str.islower': str.islower,
        'str.isupper': str.isupper,
        'str.istitle': str.istitle,
        'str.isspace': str.isspace,

        # String module functions
        'string.ascii_lowercase': string.ascii_lowercase,
        'string.ascii_uppercase': string.ascii_uppercase,
        'string.ascii_letters': string.ascii_letters,
        'string.digits': string.digits,
        'string.hexdigits': string.hexdigits,
        'string.octdigits': string.octdigits,
        'string.punctuation': string.punctuation,
        'string.whitespace': string.whitespace,
        'string.printable': string.printable,

        # Regular expression functions
        're.match': re.match,
        're.search': re.search,
        're.findall': re.findall,
        're.finditer': re.finditer,
        're.sub': re.sub,
        're.split': re.split,
        're.compile': re.compile,
    }

def load_static_eval_funcs():
    
    # get python globals
    python_globals = get_python_globals()
    
    # get realign globals
    realign_globals = get_realign_evals_utils()
    
    # get evallib globals
    evallib_globals = get_evallib_functions()
    
    # merge all globals in the order of priority
    evaluator.all_evaluators.update(python_globals)
    evaluator.all_evaluators.update(realign_globals)
    evaluator.all_evaluators.update(evallib_globals)
    
# Load static evaluator functions
load_static_eval_funcs()
