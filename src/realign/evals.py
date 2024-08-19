
import asyncio
from dataclasses import dataclass
from typing import Any, Optional, Callable, Coroutine
import inspect
import functools

import realign
from realign.config import load_yaml_settings
from realign.utils import bcolors


# ------------------------------------------------------------------------------
# EVALUATOR SETTINGS
# ------------------------------------------------------------------------------


DEFAULT_EVALUATOR_SETTINGS = {
    'weight': 1.0,
    'asserts': False,
    'repeat': None,
    'transform': None,
    'aggregate': None,
    'checker': None,
    'target': None,
    'evaluate': None
}

EVALUATOR_SETTINGS_KEYS = DEFAULT_EVALUATOR_SETTINGS.keys()

@dataclass
class EvalSettings:
    type: str
    weight: float = DEFAULT_EVALUATOR_SETTINGS['weight']
    asserts: bool = DEFAULT_EVALUATOR_SETTINGS['asserts']
    repeat: Optional[int] = DEFAULT_EVALUATOR_SETTINGS['repeat']
    transform: Optional[str] = DEFAULT_EVALUATOR_SETTINGS['transform']
    aggregate: Optional[str] = DEFAULT_EVALUATOR_SETTINGS['aggregate']
    checker: Optional[str] = DEFAULT_EVALUATOR_SETTINGS['checker']
    target: Optional[str] = DEFAULT_EVALUATOR_SETTINGS['target']
    evaluate: Optional[str] = DEFAULT_EVALUATOR_SETTINGS['evaluate']
    
    def copy(self) -> 'EvalSettings':
        return EvalSettings(
            type=self.type,
            weight=self.weight,
            repeat=self.repeat,
            asserts=self.asserts,
            transform=self.transform,
            aggregate=self.aggregate,
            checker=self.checker,
            target=self.target,
            evaluate=self.evaluate,
        )
        
    def keys(self):
        return self.__dict__.keys()
    
    def update(self, eval_settings: Any) -> None:

        """
        Update the EvalSettings instance with provided keyword arguments.
        Only updates fields that are explicitly passed.
        """
        for key, value in eval_settings.items():
            if key in self.keys():
                setattr(self, key, value)


def extract_eval_settings_and_kwargs(settings: dict[str, Any]):

    eval_kwargs = {}
    eval_settings = {}
    
    for key, value in settings.items():
        if key in EVALUATOR_SETTINGS_KEYS:
            eval_settings[key] = value
        else:
            eval_kwargs[key] = value
    
    return eval_settings, eval_kwargs

def get_eval_settings(yaml_file: Optional[str] = None,
                      eval_type: Optional[str] = None) -> tuple[dict, dict]:
    
    parsed_yaml = load_yaml_settings(yaml_file)

    if not isinstance(parsed_yaml, dict) or 'evaluators' not in parsed_yaml:
        raise ValueError("Invalid YAML structure. Expected 'evaluators' key at the root level.")

    assert isinstance(parsed_yaml['evaluators'], dict), "evaluators must be a dictionary"

    evals_settings: dict[str, EvalSettings] = dict()
    evals_kwargs: dict[str, Any] = dict()
    
    for _eval_type, settings in parsed_yaml['evaluators'].items():
        eval_settings, eval_kwargs = extract_eval_settings_and_kwargs(settings)
        evals_settings[_eval_type] = EvalSettings(type=_eval_type, **eval_settings)
        evals_kwargs[_eval_type] = eval_kwargs

    if eval_type is not None:
        if eval_type not in evals_settings:
            return EvalSettings(type=eval_type, **DEFAULT_EVALUATOR_SETTINGS), dict()
        return evals_settings[eval_type], evals_kwargs[eval_type]

    # return all evaluators
    return evals_settings, evals_kwargs

# ------------------------------------------------------------------------------
# EVALUATOR RESULT
# ------------------------------------------------------------------------------

class EvalResult:
    
    def __init__(self, 
                 score: Any, 
                 init_method: Optional[str] = None, 
                 **metadata):
        
        self.score: Any | EvalResult = score
        self.metadata: dict = metadata
        
        # determine the eval_type
        self.init_method = init_method or inspect.stack()[1].function
        
        self.eval_settings: Optional[EvalSettings] = EvalSettings(type=self.init_method)
        self.eval_kwargs: Optional[dict] = dict()
        
        # for easy access
        self.weight = self.eval_settings.weight
        
        self.func_impl: Callable = None
        self.func_args: tuple = None
        self.func_kwargs: dict = None
        self.call_depth = 0
        
        self.str = self.trace()
    
    def trace(self, indent=0):
        if 'transform' in self.init_method or 'aggregate' in self.init_method:
            method_color = bcolors.OKBLUE
        else:
            method_color = bcolors.WARNING

        result = f'{method_color}{self.init_method}{bcolors.ENDC}:\n'
        
        # function implementation
        if self.func_impl is not None:
            result += f'{" " * (indent + 4)}func_impl: {self.func_impl}\n'
        if self.func_args and len(self.func_args) > 0:
            result += f'{" " * (indent + 4)}func_args: {self.func_args}\n'
        if self.func_kwargs and len(self.func_kwargs) > 0:
            result += f'{" " * (indent + 4)}func_kwargs: {self.func_kwargs}\n'
        if self.call_depth > 0:
            result += f'{" " * (indent + 4)}call_depth: {self.call_depth}\n'
        
        result += f'{" " * (indent + 4)}score: {bcolors.FAIL}{self.score}{bcolors.ENDC}\n'
        
        if self.eval_settings:
            result += f'{" " * (indent + 4)}eval_settings:\n'
            for key in EVALUATOR_SETTINGS_KEYS:
                if hasattr(self.eval_settings, key) and (value := getattr(self.eval_settings, key)) is not None:
                    if key in DEFAULT_EVALUATOR_SETTINGS and value != DEFAULT_EVALUATOR_SETTINGS[key]:
                        result += f'{" " * (indent + 8)}{key}: {value}\n'
        
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
                    result += f'{key}: '
                    if isinstance(value, EvalResult):
                        result += f'\n{" " * (indent + 12)}{value.trace(indent + 12)}'
                    elif isinstance(value, tuple) or isinstance(value, list) and len(value) > 0:
                        result += '[\n'
                        for _, item in enumerate(value):
                            if isinstance(item, EvalResult):
                                result += f'\n{" " * (indent + 12)}{item.trace(indent + 12)}'
                            elif isinstance(item, tuple) or isinstance(item, list):
                                result += f'{" " * (indent + 12)}['
                                for _, item in enumerate(item):
                                    if isinstance(item, EvalResult):
                                        result += f'\n{" " * (indent + 16)}{item.trace(indent + 16)}'
                                    else:
                                        result += str(item)
                                result += f'\n{" " * (indent + 12)}]'
                            else:
                                result += f'\n{" " * (indent + 12)}{item}'
                        result += f'\n{" " * (indent + 8)}]'
                    else:
                        result += str(value)
                else:
                    result += str(value)
                first_item = False
            result += f'\n{" " * (indent + 4)}'
        
        self.str = result
        return result
    
    def to_dict(self) -> dict:
        return {
            'score': self.score,
            'metadata': self.metadata
        }
    
    def __str__(self):
        self.trace()
        return self.str
    
    def __repr__(self):
        self.trace()
        return self.str


# map of task_id to context
call_context: dict[str, Any] = {'depth': 0, 'calls': []}

class evaluator:

    # ------------------------------------------------------------------------------
    # STATICS / INITIALIZE
    # ------------------------------------------------------------------------------

    all_eval_settings, all_eval_kwargs = get_eval_settings(yaml_file=realign.config_path)
    all_evaluators: dict[str, 'evaluator' | Callable | Coroutine  | None] = {}
    
    def __unnamed__(self, *args, **kwargs): 
        raise NotImplementedError(f"Please decorate with an evaluator implementation.")
    
    def __new__(cls, func=None, **kwargs):
        '''Allows evaluator to be initialized in the decorator with kwargs'''
        if func is None:
            return lambda f: cls(f, **kwargs)
        return super().__new__(cls)

    def __init__(self, 
                 func: Callable = __unnamed__, 
                 eval_settings: EvalSettings = None, 
                 eval_kwargs: dict[str, Any] = None,
                 **decorator_kwargs) -> None:
        
        # set the wrapped function implementation and its name
        self.func: Callable = func
        self.name: str = func.__name__
        
        kwarg_eval_settings, kwarg_eval_kwargs = extract_eval_settings_and_kwargs(decorator_kwargs)
        
        # fetch evaluator settings/kwargs from kwargs, then yaml, then defaults
        default_evaluator = EvalSettings(type=self.name).copy()
        self.eval_settings = eval_settings or \
                                self.all_eval_settings.get(self.name, default_evaluator)
        self.eval_kwargs = eval_kwargs or \
                                self.all_eval_kwargs.get(self.name, dict())
                                
                                
        # update the settings with the provided values
        self.eval_kwargs.update(kwarg_eval_kwargs)
        self.eval_settings.update(kwarg_eval_settings)
        
        # set all_evaluators[func name] = this evaluator
        self.all_evaluators[self.name] = self
                
        # sets decorator metadata to that of wrapped function
        functools.update_wrapper(self, func)
        
        self.prev_runs = []
        
        self._prev_run = None
    
    @classmethod
    def initialize_evaluators(cls):
        '''
        After we have loaded the eval settings/kwargs, and defined the init function,
        we can initialize all_evaluators in the config.
        '''

        # instantiate all evaluators in config
        for eval_name in cls.all_eval_settings.keys():
            cls.all_evaluators[eval_name] = cls()

    # ------------------------------------------------------------------------------
    # CALLING
    # ------------------------------------------------------------------------------
    
    def apply_aggregation(self, 
                          eval_results: tuple[EvalResult] | list[EvalResult],
                          eval_scores: tuple | list, 
                          ) -> tuple[EvalResult, Any]:
        
        aggregation_expr = self.eval_settings.aggregate
        if not aggregation_expr:
            return eval_results, eval_scores
        
        locals_dict = {
            'values': eval_scores,
            'results': eval_results
        }
        
        # apply aggregation
        aggregate_score = eval(aggregation_expr, self.all_evaluators, locals_dict)
        
        init_methods = set()
        for eval_result in eval_results:
            if type(eval_result) == EvalResult:
                init_methods.add(eval_result.init_method)
        
        init_method = 'aggregate: '
        if len(init_methods) > 0:
            init_method += '-'.join(init_methods)
        
        aggregate_result = EvalResult(aggregate_score,
                                       init_method=init_method,
                                       prev_results=eval_results)
        
        return aggregate_result, aggregate_score
    
    def apply_transformation(self, 
                             eval_result: EvalResult,
                             eval_score: Any,
                             ) -> tuple[EvalResult, Any]:
        
        if not self.eval_settings.transform:
            return eval_result, eval_score
        
        transform_expr = str(self.eval_settings.transform)
        
        locals_dict = {
            'value': eval_score,
            'result': eval_result
        }
        
        # apply transformation
        transformed_score = eval(transform_expr, self.all_evaluators, locals_dict)
        
        init_method = 'transform: ' + eval_result.init_method
        
        
        transformed_result = EvalResult(transformed_score,
                                        init_method=init_method,
                                        prev_results=eval_result)
        
        # TODO: we should use call time weights here
        transformed_result.weight = self.eval_settings.weight
        
        return transformed_result, transformed_score

    def run_checker(self, 
                    eval_result: EvalResult,
                    eval_score: Any, 
                    checker: Optional[str],
                    target=None,
                    asserts=None) -> bool:
        # TODO: should we run eval() on target?
        # target = target or True
        
        if not checker:
            if not asserts:
                return eval_result, eval_score
            
            assert eval_score, f'Assertion failed: score {eval_score}'
            
            return eval_result, eval_score
        
        callable_checker = self.all_evaluators[checker]
        
        if target:
            checker_score = callable_checker(eval_score, target=target)
        else:
            checker_score = callable_checker(eval_score)
        
        if asserts:
            assert checker_score, f'Assertion failed: score {eval_score} is not in range {target}'
        
        init_method = 'checker: ' + eval_result.init_method
        
        checker_result = EvalResult(checker_score, 
                                    init_method=init_method,
                                    prev_result=eval_result)
        
        return checker_result, checker_score
    
    def async_call(self, *call_args, **call_kwargs):
        async def single_evaluation() -> tuple[EvalResult, Any]:
                
            # run the evaluator
            score = self.func(*call_args, **call_kwargs)
            
            # create result
            result = EvalResult(score, init_method=self.name)
            result.eval_settings = self.eval_settings
            result.eval_kwargs = call_kwargs
            result.func_impl = f'{self.func.__module__}.{self.func.__name__}'
            result.func_args = call_args
            result.func_kwargs = call_kwargs
            result.call_depth = call_context['depth']
            
            # transform
            transformed_result, transformed_score = self.apply_transformation(result, score)
            
            return transformed_result, transformed_score
        
        # execute repetition
        if self.eval_settings.repeat:
            # Parallel evaluation
            results, scores = asyncio.gather(
                *(
                    single_evaluation() 
                    for _ in range(self.eval_settings.repeat)
                )
            )
            results = tuple(results)
            scores = tuple(scores)
        else:
            results, scores = single_evaluation()
        
        aggregate_result, aggregate_score = self.apply_aggregation(results, scores)
            
        return aggregate_result, aggregate_score
    
    def sync_call(self, *call_args, **call_kwargs):
        
        # get call kwargs
        target = call_kwargs.get('target', self.eval_settings.target)
        asserts = call_kwargs.get('asserts', self.eval_settings.asserts)
        checker = call_kwargs.get('checker', self.eval_settings.checker)
        
        def single_evaluation() -> tuple[EvalResult, Any]:
                
            # run the evaluator
            score = self.func(*call_args, **call_kwargs)
            
            
            # create result
            result = EvalResult(score, init_method=self.name)
            # shouldn't this update with call_kwargs?
            result.eval_settings = self.eval_settings 
            result.eval_kwargs = call_kwargs
            result.func_impl = f'{self.func.__module__}.{self.func.__name__}'
            result.func_args = call_args
            result.func_kwargs = call_kwargs
            result.call_depth = call_context['depth']
            
            # transform
            transformed_result, transformed_score = self.apply_transformation(result, score)
            
            # check target on transform if aggregate not defined
            if not self.eval_settings.aggregate:
                checker_result, checker_score = self.run_checker(
                                    eval_result=transformed_result,
                                    eval_score=transformed_score, 
                                    checker=checker,
                                    target=target,
                                    asserts=asserts)
            
                return checker_result, checker_score
            
            return transformed_result, transformed_score
        
        # execute repetition
        if self.eval_settings.repeat:
            # Serial evaluation
            results, scores = zip(*tuple(
                single_evaluation() 
                for _ in range(self.eval_settings.repeat)
            ))
            results = tuple(results)
            scores = tuple(scores)
        else:
            results, scores = single_evaluation()
            
        
        # apply aggregation
        aggregate_result, aggregate_score = self.apply_aggregation(results, scores)
        
        # check target on aggregate if aggregate defined
        if self.eval_settings.aggregate:
            checker_result, checker_score = self.run_checker(
                                eval_result=aggregate_result,
                                eval_score=aggregate_score, 
                                checker=checker,
                                target=target,
                                asserts=asserts)
            
            return checker_result, checker_score
        
        return aggregate_result, aggregate_score
    
    
    def __call__(self, *args, **kwargs):
        
        # check if we are in an async context
        # TODO: async tracing not working
        def is_in_async_context():
            try:
                loop = asyncio.get_running_loop()
                return loop.is_running() and asyncio.current_task() is not None
            except RuntimeError:
                return False

        # if hasattr(self.__call__, 'task_id'):
        #     task_id = task_id
        # else:
        #     if is_in_async_context():
        #         task = asyncio.current_task()
        #         task_id = id(task)
        #     else:
        #         task_id = task_id
        
        eval_settings, eval_kwargs = extract_eval_settings_and_kwargs(kwargs)
        
        # kwargs override settings
        old_settings = self.eval_settings
        
        # merge eval_settings and eval_kwargs
        self.eval_settings.update(eval_settings)
        merged_kwargs = {**self.eval_kwargs, **eval_kwargs}
        # settings and kwargs are now final for this call
        
        call_context['depth'] += 1
        
        try:
            
            # RUN EVALUATOR
            if asyncio.iscoroutinefunction(self.func):
                results, scores = self.async_call(*args, **merged_kwargs)
            else:
                results, scores = self.sync_call(*args, **merged_kwargs)
            
            # if this is the first call, save the result.  Otherwise, add it to the list of previous runs
            if call_context['depth'] == 1:
                self._prev_run = results
            else:
                call_context['calls'].append(results)
                
        except Exception as e:
            # TODO: if there was an exception, save the result
            # results = EvalResult(e, init_method=self.name)
            raise e
            
        finally:
            call_context['calls'].reverse()
            call_context['depth'] -= 1
            
            # TODO: this is only needed for recursion
            # if call_context['depth'] == 0:
            #     if self._prev_run is not None:
            #         if isinstance(self._prev_run, (list, tuple)):
            #             if len(self._prev_run) > 0:
            #                 # add the repeats to calls
            #                 for i, result in enumerate(self._prev_run):
            #                     if isinstance(result, EvalResult):
            #                         self._prev_run[i].metadata = {
            #                             'prev_results': call_context['calls']
            #                         }
            #             else:
            #                 raise ValueError('Error tracing results: no results found!')
            #         else:    
            #             self._prev_run.metadata = {
            #                 'prev_results': call_context['calls']
            #             }
            #     else:
            #         print('Error tracing results!')

        
        self.eval_settings = old_settings
        
        return scores

    def raw(self, *args, **kwargs):
        return self.func(*args, **kwargs)


    # ------------------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------------------
    
    @property
    def settings(self) -> EvalSettings:
        return self.eval_settings

    @settings.setter
    def settings(self, value: EvalSettings):
        assert isinstance(value, EvalSettings)
        self.eval_settings = value

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.eval_kwargs

    @kwargs.setter
    def kwargs(self, value: dict):
        assert isinstance(value, dict)
        self.eval_kwargs = value

    @property
    def prev_run(self):
        if isinstance(self._prev_run, EvalResult):
            self._prev_run.trace()
        return self._prev_run   
    
    # ------------------------------------------------------------------------------
    # ACCESSORS
    # ------------------------------------------------------------------------------
    
    @classmethod
    def _validate_key(cls, key: str | Callable | None) -> str:
        if isinstance(key, str):
            return key
        elif isinstance(key, Callable):
            return key.__name__
        else:
            raise KeyError(f"Invalid key type: {type(key)}")

    @classmethod
    def __class_getitem__(cls, key: str | Callable) -> 'evaluator':
        key = cls._validate_key(key)
        if key in cls.all_evaluators:
            return cls.all_evaluators[key]
        else:
            raise KeyError(f"Key '{key}' not found in evaluators.")

    @classmethod
    def __class_setitem__(cls, key, value):
        key = cls._validate_key(key)
        cls.all_evaluators[key] = value

    @classmethod
    def __class_delitem__(cls, key):
        key = cls._validate_key(key)
        del cls.all_evaluators[key]

    # ------------------------------------------------------------------------------
    # ASYNC UTILS
    # ------------------------------------------------------------------------------
    
    @classmethod
    async def gather(cls, *evaluators):
        return await asyncio.gather(*(evaluator() for evaluator in evaluators))

    def a_sync(self, *args, **kwargs):
        if asyncio.iscoroutinefunction(self.func):
            return self(*args, **kwargs)
        else:
            async def wrapper(*args, **kwargs):
                return self(*args, **kwargs)
            return wrapper(*args, **kwargs)

    def sync(self, *args, **kwargs):
        if asyncio.iscoroutinefunction(self.func):
            if asyncio.get_event_loop().is_running():
                return asyncio.create_task(self(*args, **kwargs))
            else:
                return asyncio.run(self(*args, **kwargs))
        else:
            return self(*args, **kwargs)
    

# ------------------------------------------------------------------------------
# INSTANTIATE EVALLIB
# ------------------------------------------------------------------------------

# instantiate all decorated evaluators in evallib
from realign import evallib


# ------------------------------------------------------------------------------
# PRINTING
# ------------------------------------------------------------------------------

def print_evals(evals: list[EvalResult]):
    print(bcolors.WARNING)
    for e in evals:
        print(e)
        print("- " * 50)
    print(bcolors.ENDC)


