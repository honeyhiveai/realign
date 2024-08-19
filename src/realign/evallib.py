from typing import Callable
import inspect

from realign.evals import evaluator, EvalResult, EvalSettings


# ------------------------------------------------------------------------------
# LIBRARY OF EVALUATORS
# ------------------------------------------------------------------------------

import itertools
import math
import random
import string
import re

@evaluator
def weighted_sum(values, results: list[EvalResult]):
    
    weighted_sum = 0
    for value, result in zip(values, results):
        print(value, result.weight)
        weighted_sum += float(value) * result.weight
    
    return weighted_sum

@evaluator
def weighted_mean(values, results: list[EvalResult]):
    return weighted_sum(values, results) / sum(result.weight for result in results)

@evaluator
def identity(x) -> EvalResult:
    return EvalResult(x)

@evaluator
def pymean(x) -> EvalResult:
    score = sum(x)/len(x)
    return EvalResult(score)

@evaluator
def npmean(x) -> EvalResult:
    try:
        import numpy as np
    except ImportError:
        print("Please install numpy to use npmean")
        raise
    
    score = np.mean(x)
    score += 7
    
    return EvalResult(score)

@evaluator
def zero(x) -> EvalResult:
    return EvalResult(0)

@evaluator
def fourtytwo(x):
    # print('getting settings inside fourtytwo', fourtytwo.settings)
    return 42

@evaluator
def numrange(x):
    '''Evaluator for checking if a numeric value is within a passing range'''
    
    assert x, 'numrange requires an argument'
    
    # For the numrange evaluator, target is the target numeric range
    num_range = numrange.settings.target
    
    if num_range is None:
        return True
    
    def in_num_interval(num_interval: tuple | list, x):
        '''Checks if x is between two numbers in num_interval, inclusive of the bounds'''
        
        assert type(num_interval) in [tuple, list] and len(num_interval) == 2, \
            'pass_range must be a tuple or list of length 2'

        left, right = num_interval
        
        def check(x):
            return left <= x <= right

        # if score is iterable, check all elements
        if hasattr(x, '__iter__'):
            return all(check(x_i) for x_i in x)

        return check(x)
    
    def is_equal(num, x):
        '''Checks if x is equal to num'''
        
        def check(x):
            return x == num

        # if score is iterable, check all elements
        if hasattr(x, '__iter__'):
            return all(check(x_i) for x_i in x)

        return check(x)

    def in_str_interval(str_interval, x):
        '''Checks if x is within the interval defined by str_interval'''
        
        # Remove whitespace from the interval string
        str_interval = str_interval.replace(" ", "")

        # Extract the bounds and inclusivity from the interval string
        left_bound, right_bound = str_interval.split(",")
        left_inclusive = left_bound[0] == "["
        right_inclusive = right_bound[-1] == "]"
        left = float(left_bound[1:]) if left_bound[1:] else None
        right = float(right_bound[:-1]) if right_bound[:-1] else None

        def check_interval(x):
            # Check if x is within the interval
            if left is None and right is None:
                return True
            elif left is None:
                if right_inclusive:
                    return x <= right
                else:
                    return x < right
            elif right is None:
                if left_inclusive:
                    return left <= x
                else:
                    return left < x
            else:
                assert left <= right, f'invalid target range {str_interval}'
                if left_inclusive and right_inclusive:
                    return left <= x <= right
                elif left_inclusive and not right_inclusive:
                    return left <= x < right
                elif not left_inclusive and right_inclusive:
                    return left < x <= right
                else:
                    return left < x < right

        # if score is iterable, check all elements
        if hasattr(x, '__iter__'):
            return all(check_interval(x_i) for x_i in x)
        
        return check_interval(x)
        
    in_interval = None
    if type(num_range) == str:
        in_interval = in_str_interval
    elif type(num_range) in [list, tuple]:
        in_interval = in_num_interval
    elif type(num_range) in [int, float]:
        in_interval = is_equal
    else:
        raise ValueError('num_range must be a string, list, tuple, or int/float')
        
    score = in_interval(num_range, x)
    
    return score



# ------------------------------------------------------------------------------
# LOAD ALL EVALUATORS 
# ------------------------------------------------------------------------------

EXCLUDE_GLOBALS = [
    'get_evallib_functions',
    'get_realign_utils',
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
