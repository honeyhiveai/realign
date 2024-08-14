import json
import hashlib
import operator
import inspect
import functools
from typing import Any, Optional, Callable
from dataclasses import dataclass

from realign.config import EvalSettings, EVALUATOR_SETTINGS_KEYS

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@dataclass
class OpenAIMessage:
    role: str
    content: str | dict[str, str]

    def __dict__(self):
        return {
            'role': str(self.role),
            'content': str(self.content)
        }

@dataclass
class RunData:
    final_state: Any
    run_id: Optional[int] = None
    
    def __dict__(self):
        return {
            'run_id': self.run_id,
            'final_state': self.final_state
        }
    
    def __repr__(self) -> str:
        return str(self.__dict__())
    
    def compute_hash(self, hash_algorithm='sha256'):
        """
        Compute a hash of a RunData.
        
        :param obj: The object to hash
        :param hash_algorithm: The hash algorithm to use (default is 'sha256')
        :return: A hexadecimal string representation of the hash
        """
        # Convert the object to a JSON string
        json_string = json.dumps(self.__dict__(), sort_keys=True, default=str)
        
        # Create a hash object with the specified algorithm
        hash_object = hashlib.new(hash_algorithm)
        
        # Update the hash object with the JSON string (encoded to bytes)
        hash_object.update(json_string.encode('utf-8'))
        
        # Return the hexadecimal representation of the hash
        return hash_object.hexdigest()    


class EvalResult:
    
    def __init__(self, score, init_method=None, **metadata):
        self.score: Any | EvalResult = score
        self.metadata: dict = metadata
        
        # determine the eval_type
        self.init_method = init_method or inspect.stack()[1].function
        
        self.eval_settings: Optional[EvalSettings] = EvalSettings(type=self.init_method)
        self.eval_kwargs: Optional[dict] = dict()
        
        self.weight = self.eval_settings.weight
        
        self.str = self.trace()
    
    def trace(self, indent=0):
        if self.init_method in ['apply_transformation', 'apply_aggregation']:
            method_color = bcolors.OKBLUE
        else:
            method_color = bcolors.WARNING

        result = f'{method_color}{self.init_method}{bcolors.ENDC}:\n'
        result += f'{" " * (indent + 4)}score: {bcolors.FAIL}{self.score}{bcolors.ENDC}\n'
        
        if self.eval_settings:
            result += f'{" " * (indent + 4)}eval_settings:\n'
            for key in EVALUATOR_SETTINGS_KEYS:
                if hasattr(self.eval_settings, key) and getattr(self.eval_settings, key) is not None:
                    result += f'{" " * (indent + 8)}{key}: {getattr(self.eval_settings, key)}\n'
        
        if self.eval_kwargs:
            result += f'{" " * (indent + 4)}eval_kwargs:\n'
            for key, value in self.eval_kwargs.items():
                if value is not None:
                    result += f'{" " * (indent + 8)}{key}: {value}\n'
        
        if self.metadata:
            result += f'\n{" " * (indent + 4)}metadata:'
            first_item = True
            for key, value in self.metadata.items():
                if not first_item:
                    result += ','
                result += f'\n{" " * (indent + 8)}'
                if key in ('prev_result', 'prev_results'):
                    if isinstance(value, EvalResult):
                        result += value.trace(indent + 8)
                    elif isinstance(value, tuple) or isinstance(value, list):
                        result += '['
                        for _, item in enumerate(value):
                            if isinstance(item, EvalResult):
                                result += f'\n{" " * (indent + 12)}{item.trace(indent + 12)}'
                            else:
                                result += f'\n{" " * (indent + 12)}{item}'
                        result += f'\n{" " * (indent + 8)}]'
                    else:
                        result += str(value)
                else:
                    result += str(value)
                first_item = False
            result += f'\n{" " * (indent + 4)}'
        
        return result
    
    def __bool__(self) -> bool:
        return bool(self.score)

    def _operation(self, other: Any, op: Callable) -> 'EvalResult':
        if isinstance(other, EvalResult):
            other = other.score

        return EvalResult(op(self.score, other), 
                              init_method=self.init_method, 
                              **self.metadata)

    def _r_operation(self, other: Any, op: Callable) -> 'EvalResult':
        return EvalResult(op(other, self.score), 
                          init_method=self.init_method,
                          **self.metadata)

    __add__ = __radd__ = lambda self, other: self._operation(other, operator.add)
    __sub__ = lambda self, other: self._operation(other, operator.sub)
    __rsub__ = lambda self, other: self._r_operation(other, operator.sub)
    __mul__ = __rmul__ = lambda self, other: self._operation(other, operator.mul)
    __truediv__ = lambda self, other: self._operation(other, operator.truediv)
    __rtruediv__ = lambda self, other: self._r_operation(other, operator.truediv)
    __floordiv__ = lambda self, other: self._operation(other, operator.floordiv)
    __rfloordiv__ = lambda self, other: self._r_operation(other, operator.floordiv)
    __mod__ = lambda self, other: self._operation(other, operator.mod)
    __rmod__ = lambda self, other: self._r_operation(other, operator.mod)
    __pow__ = lambda self, other: self._operation(other, operator.pow)
    __rpow__ = lambda self, other: self._r_operation(other, operator.pow)

    __neg__ = lambda self: EvalResult(-self.score, 
                                      init_method=self.init_method, **self.metadata)
    __pos__ = lambda self: EvalResult(+self.score, 
                                      init_method=self.init_method,
                                      **self.metadata)
    __abs__ = lambda self: EvalResult(abs(self.score), 
                                      init_method=self.init_method,
                                      **self.metadata)

    def __int__(self) -> int:
        return int(self.score)

    def __float__(self) -> float:
        return float(self.score)

    def __repr__(self) -> str:
        return f"{self.score}"

    def __eq__(self, other: Any) -> bool:
        return self.score == (other.score if isinstance(other, EvalResult) else other)

    def __lt__(self, other: Any) -> bool:
        return self.score < (other.score if isinstance(other, EvalResult) else other)

    __le__ = lambda self, other: self < other or self == other
    __gt__ = lambda self, other: not (self <= other)
    __ge__ = lambda self, other: not (self < other)


def evaluator(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> EvalResult:
        result = func(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return EvalResult(result[0], init_method=func.__name__, **result[1])
        return EvalResult(result, init_method=func.__name__)

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> EvalResult:
        result = await func(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return EvalResult(result[0], init_method=func.__name__, **result[1])
        return EvalResult(result, init_method=func.__name__)

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return wrapper