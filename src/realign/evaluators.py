import asyncio
from dataclasses import dataclass, field, fields
from typing import Any, Optional, Callable, Coroutine
import inspect
import functools
import json

from realign.utils import bcolors


# ------------------------------------------------------------------------------
# EVALUATOR SETTINGS
# ------------------------------------------------------------------------------



EVALUATOR_SETTINGS_KEYS = [
    'wraps', 
    'weight', 
    'asserts', 
    'repeat', 
    'transform', 
    'aggregate', 
    'checker', 
    'target', 
    'evaluate'
]


@dataclass
class EvalSettings:
    name: str  # config key
    wraps: Optional[str | dict] = None
    weight: float = None
    asserts: bool = None
    repeat: Optional[int] = None
    transform: Optional[str] = None
    aggregate: Optional[str] = None
    checker: Optional[str] = None
    target: Optional[str] = None
    evaluate: Optional[str] = None

    def copy(self) -> "EvalSettings":
        return EvalSettings(
            name=self.name,
            wraps=self.wraps,
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

    # TODO: settings update should replace instead of merge
    def update(self, eval_settings: Any | None) -> None:
        if eval_settings is None:
            return
        if isinstance(eval_settings, dict):
            update_dict = eval_settings
        elif isinstance(eval_settings, EvalSettings):
            update_dict = eval_settings.__dict__
        else:
            raise TypeError(
                "eval_settings must be either a dictionary or an EvalSettings instance. Got {}".format(
                    type(eval_settings)
                )
            )

        valid_fields = {f.name for f in fields(self)}

        for key, value in update_dict.items():
            if key not in valid_fields:
                raise ValueError(f"Invalid field name: {key}")
            if value is not None:
                setattr(self, key, value)
    
    @staticmethod
    def extract_eval_settings_and_kwargs(settings: dict[str, Any] | None):

        eval_kwargs = dict()
        eval_settings = dict()

        if settings is not None:
            for key, value in settings.items():
                if key in EVALUATOR_SETTINGS_KEYS:
                    eval_settings[key] = value
                else:
                    eval_kwargs[key] = value

        return eval_settings, eval_kwargs

    def __str__(self) -> str:
        dict_str = {k: str(v) for k, v in self.__dict__.items() if v is not None}
        return json.dumps(dict_str, indent=4).replace('"', "")

    def dict(self) -> dict:
        ret_dict = self.__dict__
        ret_dict.pop('name', None)
        return ret_dict

@dataclass
class EvaluatorSettings:
    name: str
    
    # base default settings
    default_settings: EvalSettings = None
    default_kwargs: dict = field(default_factory=dict)
    
    # settings from defaults.yaml
    defaults_yaml_settings: EvalSettings = None
    defaults_yaml_kwargs: dict = field(default_factory=dict)
    
    # settings from evaluator init
    init_settings: EvalSettings = None
    init_kwargs: dict = field(default_factory=dict)
    
    # settings from decorator kwargs
    deco_settings: EvalSettings = None
    deco_kwargs: dict = field(default_factory=dict)
    
    # settings from config file
    config_settings: EvalSettings = None
    config_kwargs: dict = field(default_factory=dict)
    
    # settings from runtime
    explicit_settings: EvalSettings = None
    explicit_kwargs: dict = field(default_factory=dict)
    

    def __post_init__(self):
        self.default_settings = EvalSettings(name=self.name)
        self.defaults_yaml_settings = EvalSettings(name=self.name)
        self.init_settings = EvalSettings(name=self.name)
        self.deco_settings = EvalSettings(name=self.name)
        self.config_settings = EvalSettings(name=self.name)
        self.explicit_settings = None # this must be set at runtime
        
        # set defaults
        self.default_settings.asserts = False
        self.default_settings.weight = 1.0

    def resolve_settings(self, settings: EvalSettings | None = None):
        if self.explicit_settings:
            return self.explicit_settings
        
        final_settings: EvalSettings = self.default_settings
        final_settings.update(self.defaults_yaml_settings)
        final_settings.update(self.init_settings)
        final_settings.update(self.config_settings)
        final_settings.update(self.deco_settings)
        
        if settings is not None:
            final_settings.update(settings)
        
        return final_settings
    
    def resolve_kwargs(self, kwargs: dict | None = None):
        if self.explicit_kwargs:
            return self.explicit_kwargs
        
        final_kwargs: dict = self.default_kwargs.copy()
        final_kwargs.update(self.defaults_yaml_kwargs)
        final_kwargs.update(self.init_kwargs)
        final_kwargs.update(self.config_kwargs)
        final_kwargs.update(self.deco_kwargs)
        
        if kwargs is not None:
            final_kwargs.update(kwargs)

        return final_kwargs
        
# ------------------------------------------------------------------------------
# EVALUATOR RESULT
# ------------------------------------------------------------------------------


class EvalResult:

    def __init__(self, score: Any, init_method: Optional[str] = None, **metadata):

        self.score: Any | EvalResult = score
        self.metadata: dict = metadata

        # determine the eval_type
        self.init_method = init_method or inspect.stack()[1].function

        self.eval_settings: Optional[EvalSettings] = EvalSettings(name=self.init_method)
        self.eval_kwargs: Optional[dict] = dict()

        # for easy access
        self.weight = self.eval_settings.weight

        self.func_impl: Callable = None
        self.func_args: tuple = None
        self.func_kwargs: dict = None
        self.call_depth = 0

        self.str = self.trace()

    def trace(self, indent=0):
        if "transform" in self.init_method or "aggregate" in self.init_method:
            method_color = bcolors.OKBLUE
        else:
            method_color = bcolors.WARNING

        result = f"{method_color}{self.init_method}{bcolors.ENDC}:\n"

        # function implementation
        if self.func_impl is not None:
            result += f'{" " * (indent + 4)}func_impl: {self.func_impl}\n'
        if self.func_args and len(self.func_args) > 0:
            if len(self.func_args) == 1:
                func_args = self.func_args[0]
            result += f'{" " * (indent + 4)}func_args: \n{bcolors.OKGREEN}{func_args}{bcolors.ENDC}\n'
        if self.func_kwargs and len(self.func_kwargs) > 0:
            result += f'{" " * (indent + 4)}func_kwargs: {self.func_kwargs}\n'
        if self.call_depth > 0:
            result += f'{" " * (indent + 4)}call_depth: {self.call_depth}\n'

        result += (
            f'{" " * (indent + 4)}score: {bcolors.FAIL}{self.score}{bcolors.ENDC}\n'
        )

        if self.eval_settings:
            result += f'{" " * (indent + 4)}eval_settings:\n'
            for key in EVALUATOR_SETTINGS_KEYS:
                if (
                    hasattr(self.eval_settings, key)
                    and (value := getattr(self.eval_settings, key)) is not None
                ):
                    if (
                        key in EVALUATOR_SETTINGS_KEYS
                        and value is not None
                    ):
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
                    result += ","
                result += f'\n{" " * (indent + 8)}'
                if key in ("prev_result", "prev_results"):
                    result += f"{key}: "
                    if isinstance(value, EvalResult):
                        result += f'\n{" " * (indent + 12)}{value.trace(indent + 12)}'
                    elif (
                        isinstance(value, tuple)
                        or isinstance(value, list)
                        and len(value) > 0
                    ):
                        result += "[\n"
                        for _, item in enumerate(value):
                            if isinstance(item, EvalResult):
                                result += (
                                    f'\n{" " * (indent + 12)}{item.trace(indent + 12)}'
                                )
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
        return {"score": self.score, "metadata": self.metadata}

    def __str__(self):
        self.trace()
        return self.str

    def __repr__(self):
        self.trace()
        return self.str

    def copy(self) -> "EvalResult":
        copy_result = EvalResult(score=self.score,
                          init_method=self.init_method,
                          **self.metadata)
        copy_result.str = self.trace()
        return copy_result

# map of task_id to context
call_context: dict[str, Any] = {"depth": 0, "calls": []}


class EvaluatorMeta(type):
    
    def __getattribute__(cls, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return cls.__class_getitem__(name)
        

class evaluator(metaclass=EvaluatorMeta):

    # ------------------------------------------------------------------------------
    # STATICS / INITIALIZE
    # ------------------------------------------------------------------------------

    all_evaluators: dict[str, "evaluator" | Callable | Coroutine | "aevaluator"] = dict()
    all_evaluator_settings: dict[str, EvaluatorSettings] = dict()

    def __unnamed__(self, *args, **kwargs):
        raise NotImplementedError(f"Please decorate with an evaluator implementation.")

    def __new__(
        cls, 
        func=None, 
        eval_settings=None, 
        eval_kwargs=None, 
        **deco_settings_kwargs
    ) -> "evaluator":
        """Allows evaluator to be initialized in the decorator with kwargs"""
        if func is None:
            return lambda f: cls(f, eval_settings, eval_kwargs, **deco_settings_kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        func: Callable = __unnamed__,
        eval_settings: EvalSettings | None = None,
        eval_kwargs: dict[str, Any] | None = None,
        **deco_settings_kwargs,
    ) -> None:
        
        # set the wrapped function implementation and its name
        self.func: Callable = func

        # set the evaluator name
        self.name: str = func.__name__ if hasattr(func, "__name__") else str(func)
        
        # register all_evaluators[func name] = this evaluator
        self.all_evaluators[self.name] = self
        
        # initialize the evaluator settings
        if self.name not in evaluator.all_evaluator_settings:
            evaluator.all_evaluator_settings[self.name] = EvaluatorSettings(self.name)

        # get the settings
        evaluator_settings = evaluator.all_evaluator_settings[self.name]
            
        # init settings
        if eval_settings:
            eval_settings.name = self.name
        else:
            eval_settings = EvalSettings(name=self.name)
        
        evaluator_settings.init_settings = eval_settings
        evaluator_settings.init_kwargs = eval_kwargs or {}
        
        # decorator kwargs
        kwarg_eval_settings, kwarg_eval_kwargs = EvalSettings.extract_eval_settings_and_kwargs(
            deco_settings_kwargs
        )
        kwarg_eval_settings['name'] = self.name
        evaluator_settings.deco_settings = EvalSettings(**kwarg_eval_settings)
        evaluator_settings.deco_kwargs = kwarg_eval_kwargs
        
        self.explicit_config = None

        # sets decorator metadata to that of wrapped function
        functools.update_wrapper(self, func)
        functools.update_wrapper(func, self)

        # traces
        self.prev_runs = []
        self._prev_run = None

    # ------------------------------------------------------------------------------
    # AGGREGATION
    def pre_apply_aggregation(
        self,
        eval_results: tuple[EvalResult] | list[EvalResult],
        eval_scores: tuple | list,
        final_settings: EvalSettings,
    ) -> tuple[EvalResult, Any] | Coroutine:

        locals_dict = {
            "values": eval_scores, 
            "results": eval_results
        }

        aggregation_expr = str(final_settings.aggregate)

        # apply aggregation
        aggregate_score = eval(
            aggregation_expr, 
            evaluator.all_evaluators, 
            locals_dict
        )

        return aggregate_score

    def post_apply_aggregation(
        self,
        eval_results: tuple[EvalResult] | list[EvalResult] | EvalResult,
        aggregate_score: Any,
        final_settings: EvalSettings,
    ):
        init_methods = set()

        # if no repetitions, we will only have one eval result
        if isinstance(eval_results, EvalResult):
            init_methods.add(eval_results.init_method)
        else:
            for eval_result in eval_results:
                if isinstance(eval_result, EvalResult):
                    init_methods.add(eval_result.init_method)

        init_method = "aggregate: "
        if len(init_methods) > 0:
            init_method += "-".join(init_methods)

        aggregate_result = EvalResult(
            aggregate_score, 
            init_method=init_method, 
            prev_results=eval_results
        )

        return aggregate_result

    def sync_apply_aggregation(
        self,
        eval_results: tuple[EvalResult] | list[EvalResult] | EvalResult,
        eval_scores: tuple | list | Any,
        final_settings: EvalSettings,
    ) -> tuple[EvalResult, Any]:

        if not final_settings.aggregate:
            return eval_results, eval_scores

        aggregate_score = self.pre_apply_aggregation(
            eval_results, 
            eval_scores, 
            final_settings
        )
        
        aggregate_score = evaluator.resolve_pipeline(aggregate_score, eval_scores)
            
        aggregate_result = self.post_apply_aggregation(
            eval_results, 
            aggregate_score, 
            final_settings
        )

        return aggregate_result, aggregate_score

    async def async_apply_aggregation(
        self,
        eval_results: tuple[EvalResult] | list[EvalResult],
        eval_scores: tuple | list,
        final_settings: EvalSettings,
    ) -> tuple[EvalResult, Any]:

        if not final_settings.aggregate:
            return eval_results, eval_scores

        aggregate_score = self.pre_apply_aggregation(
            eval_results, 
            eval_scores, 
            final_settings
        )

        aggregate_score = await evaluator.aresolve_pipeline(aggregate_score, eval_scores)

        aggregate_result = self.post_apply_aggregation(
            eval_results, 
            aggregate_score, 
            final_settings
        )

        return aggregate_result, aggregate_score
    # ------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------
    # TRANSFORMATION
    def pre_apply_transformation(
        self, 
        eval_result: EvalResult, 
        eval_score: Any,
        final_settings: EvalSettings,
    ):

        transform_expr = str(final_settings.transform)

        locals_dict = {
            "value": eval_score, 
            "result": eval_result
        }

        # apply transformation
        transformed_score = eval(
            transform_expr, 
            evaluator.all_evaluators, 
            locals_dict
        )

        return transformed_score

    def post_apply_transformation(
        self, 
        eval_result: EvalResult, 
        transformed_score: Any,
        final_settings: EvalSettings,
    ):
        init_method = "transform: " + eval_result.init_method

        transformed_result = EvalResult(
            transformed_score, 
            init_method=init_method, 
            prev_result=eval_result
        )
        
        transformed_result.weight = final_settings.weight

        return transformed_result

    async def async_apply_transformation(
        self,
        eval_result: EvalResult,
        eval_score: Any,
        final_settings: EvalSettings,
    ) -> tuple[EvalResult, Any]:

        if not final_settings.transform:
            return eval_result, eval_score

        transformed_score = self.pre_apply_transformation(
            eval_result, 
            eval_score, 
            final_settings
        )

        transformed_score = await evaluator.aresolve_pipeline(transformed_score, eval_score)

        transformed_result = self.post_apply_transformation(
            eval_result, 
            transformed_score, 
            final_settings
        )

        return transformed_result, transformed_score

    def sync_apply_transformation(
        self,
        eval_result: EvalResult,
        eval_score: Any,
        final_settings: EvalSettings,
    ) -> tuple[EvalResult, Any]:

        if not final_settings.transform:
            return eval_result, eval_score

        transformed_score = self.pre_apply_transformation(
            eval_result, eval_score, final_settings
        )
        
        transformed_score = evaluator.resolve_pipeline(transformed_score, eval_score)

        transformed_result = self.post_apply_transformation(
            eval_result, transformed_score, final_settings
        )

        return transformed_result, transformed_score
    # ------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------
    # CHECKER
    def pre_run_checker(
        self,
        eval_result: EvalResult,
        eval_score: Any,
        final_settings: EvalSettings,
    ) -> bool:

        checker_expr = str(final_settings.checker)

        locals_dict = {
            "value": eval_score, 
            "result": eval_result, 
            "target": final_settings.target
        }

        # evaluate checker
        checker_score = eval(
            checker_expr, 
            evaluator.all_evaluators, 
            locals_dict
        )

        return checker_score

    def post_run_checker(
        self,
        eval_result: EvalResult,
        eval_score: Any,
        final_settings: EvalSettings,
        checker_score: Any = None,
    ) -> bool:

        if final_settings.asserts:
            assert (
                checker_score
            ), f"Assertion failed: score {eval_score} is not in range {final_settings.target}"

        init_method = "checker: " + eval_result.init_method

        checker_result = EvalResult(
            checker_score, 
            init_method=init_method, 
            prev_result=eval_result
        )

        return checker_result

    def run_checker(
        self,
        eval_result: EvalResult,
        eval_score: Any,
        final_settings: EvalSettings,
    ) -> bool:

        if not final_settings.checker:
            if not final_settings.asserts:
                return eval_result, eval_score

            assert eval_score, f"Assertion failed: score {eval_score} is not in range {final_settings.target}"

            return eval_result, eval_score

        checker_score = self.pre_run_checker(
            eval_result, 
            eval_score, 
            final_settings
        )
        
        checker_score = evaluator.resolve_pipeline(checker_score, eval_score, final_settings.target)

        checker_result = self.post_run_checker(
            eval_result, 
            eval_score, 
            final_settings, 
            checker_score
        )

        return checker_result, checker_score

    async def arun_checker(
        self,
        eval_result: EvalResult,
        eval_score: Any,
        final_settings: EvalSettings,
    ) -> bool:

        if not final_settings.checker:
            if not final_settings.asserts:
                return eval_result, eval_score

            assert eval_score, f"Assertion failed: score {eval_score} is not in range {final_settings.target}"

            return eval_result, eval_score

        checker_score = self.pre_run_checker(
            eval_result, 
            eval_score,
            final_settings
        )

        checker_score = await evaluator.aresolve_pipeline(
            checker_score, 
            eval_score, 
            final_settings.target
        )

        checker_result = self.post_run_checker(
            eval_result, 
            eval_score, 
            final_settings, 
            checker_score
        )

        return checker_result, checker_score
    # ------------------------------------------------------------------------------
    
    
    # ------------------------------------------------------------------------------
    # WRAPPING
    @staticmethod
    def parse_wraps(wraps: str | dict | None | Any):
        if wraps is None:
            return None, {}
        
        if isinstance(wraps, str):
            return wraps, {}
        elif isinstance(wraps, dict):
            # assert that there is a single key of type str
            assert len(wraps) == 1 and isinstance(list(wraps.keys())[0], str), \
                "wraps must be a single key of type str"
            
            wrapped_eval_name = list(wraps.keys())[0]
            wrapped_eval_settings_kwargs = wraps[wrapped_eval_name]
            return wrapped_eval_name, wrapped_eval_settings_kwargs
        else:
            raise ValueError(f"Invalid wraps type: {type(wraps)}")
    
    @staticmethod
    def create_wrapper(
        base_callable: 'evaluator',
        wrapped_eval_settings: EvalSettings,
        wrapped_eval_kwargs: dict,
        wrapper_name: str,
    ) -> Callable:
        """
        Create a wrapper function for an evaluator, given the base evaluator,
        the wrapped evaluator settings, the wrapped evaluator kwargs, and the wrapper name.
        
        The wrapped_eval / base_callable's settings and kwargs update any previous settings and kwargs.
        The wrapper's settings do NOT update the wrapped evaluator's settings.
        The wrapper's kwargs DO update the wrapped evaluator's kwargs.
        
        The final settings and kwargs are passed into the wrapped evaluator during calltime.
        Due to the ordering of the dict unpacking, the wrapper's kwargs will update the 
        wrapped evaluator's kwargs. The settings are also passed as kwargs into the base callable.

        Args:
            base_callable (evaluator): The base evaluator to be wrapped.
            wrapped_eval_settings (EvalSettings): Settings for the wrapped evaluator.
            wrapped_eval_kwargs (dict): Additional keyword arguments for the wrapped evaluator.
            wrapper_name (str): Name for the wrapper function.

        Returns:
            Callable: A wrapper function that calls the base evaluator with the provided settings and arguments.
        """
        
        
        base_callable_settings = wrapped_eval_settings.copy()
        base_callable_kwargs = wrapped_eval_kwargs.copy()
        
        if asyncio.iscoroutinefunction(base_callable.func):
            async def afunc(*args, **kwargs):
                return await base_callable(
                    *args,
                    ** {
                        **base_callable_settings.dict(),
                        **base_callable_kwargs,
                        **kwargs
                    }
                )
            afunc.__name__ = wrapper_name
            return afunc
        
        def func(*args, **kwargs):
            return base_callable(
                *args,
                ** {
                    **base_callable_settings.dict(),
                    **base_callable_kwargs,
                    **kwargs
                }
            )
        func.__name__ = wrapper_name
        return func
    
    @staticmethod
    def create_wrapped_evaluator(
        evaluator_settings: EvaluatorSettings
    ) -> None:
        
        wrapper_name = evaluator_settings.name
        wrapper_settings = evaluator_settings.resolve_settings()
        wrapper_kwargs = evaluator_settings.resolve_kwargs()
        
        # parse the wrapped evaluator
        wrapped_eval_name, wrapped_eval_settings_kwargs = evaluator.parse_wraps(
            wrapper_settings.wraps
        )
        assert isinstance(wrapped_eval_name, str), \
            f"wrapped evaluator name must be a string but got: {type(wrapped_eval_name)}"
        
        wrapped_eval_settings, wrapped_eval_kwargs = \
            EvalSettings.extract_eval_settings_and_kwargs(wrapped_eval_settings_kwargs)
        
        # create the wrapped evaluator settings if not already created
        # this can happen if the wrapped evaluator is not defined in any config file
        if wrapped_eval_name not in evaluator.all_evaluator_settings:
            evaluator.all_evaluator_settings[wrapped_eval_name] = EvaluatorSettings(name=wrapped_eval_name)
        
        
        # get the wrapped evaluator. It must be a registered evaluator
        base_callable = eval(
            wrapped_eval_name, 
            evaluator.all_evaluators,
        )
        
        # TODO: if the base callable is not a Callable but
        # not an evaluator, we need to wrap it as well
        assert isinstance(base_callable, evaluator)
        
        
        
        # update the wrapped evaluator settings with the wrapper settings
        final_wrapped_eval_settings = evaluator.all_evaluator_settings[wrapped_eval_name].resolve_settings().copy()
        final_wrapped_eval_settings.update(wrapped_eval_settings)
        
        # update the wrapped evaluator kwargs with the wrapper kwargs
        final_wrapped_eval_kwargs = evaluator.all_evaluator_settings[wrapped_eval_name].resolve_kwargs().copy()
        final_wrapped_eval_kwargs.update(wrapped_eval_kwargs)
        
        # the wrapper's kwargs merge with the wrapped evaluator's kwargs,
        # but the settings don't.
        final_wrapped_eval_kwargs.update(wrapper_kwargs)
        
        
        if asyncio.iscoroutinefunction(base_callable.func):
            afunc: Coroutine = evaluator.create_wrapper(
                base_callable,
                final_wrapped_eval_settings,
                final_wrapped_eval_kwargs,
                wrapper_name,
            )
            
            # initialize and register the wrapper eval
            aevaluator(func=afunc)
            
        else:
            # make the wrapper eval
            func: Callable = evaluator.create_wrapper(
                base_callable,
                final_wrapped_eval_settings,
                final_wrapped_eval_kwargs,
                wrapper_name,
            )

            # initialize and register the wrapper eval
            evaluator(func=func)
    # ------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------
    # CALLING
    @staticmethod
    async def aresolve_pipeline(score, *args, **kwargs):
        if asyncio.iscoroutinefunction(score) or isinstance(score, aevaluator):
            score = await score(*args, **kwargs)
        elif isinstance(score, Callable):
            score = score(*args, **kwargs)

        return score
    
    @staticmethod
    def resolve_pipeline(score, *args, **kwargs):
        # string evaluated to a function
        if isinstance(score, Callable) or isinstance(score, evaluator):
            score = score(*args, **kwargs)
        return score
    
    def get_final_settings_and_kwargs(self, call_kwargs):
        eval_settings, eval_kwargs = EvalSettings.extract_eval_settings_and_kwargs(call_kwargs)
        explicit_settings, explicit_kwargs = EvalSettings.extract_eval_settings_and_kwargs(self.explicit_config)
        
        # calltime copy
        final_settings = self.settings.copy()
        final_settings.update(eval_settings)
        final_settings.update(explicit_settings)
        
        final_kwargs = self.kwargs.copy()
        final_kwargs.update(eval_kwargs)
        final_kwargs.update(explicit_kwargs)
        
        return final_settings, final_kwargs

    async def async_call(self, *call_args, **call_kwargs):
        
        final_settings, final_kwargs = self.get_final_settings_and_kwargs(call_kwargs)

        async def asingle_evaluation() -> tuple[EvalResult, Any]:

            # run the evaluator
            score = await self.func(*call_args, **final_kwargs)
            result = EvalResult(
                score=score, 
                init_method=self.name,
                eval_settings=final_settings,
                eval_kwargs=final_kwargs,
                func_impl=f"{self.func.__module__}.{self.func.__name__}",
                func_args=call_args,
                func_kwargs=call_kwargs,
                call_depth=call_context["depth"]
            )

            # transform
            transformed_result, transformed_score = (
                await self.async_apply_transformation(
                    result, 
                    score, 
                    final_settings
                )
            )

            # check target on transform if aggregate not defined
            if not final_settings.aggregate:
                checker_result, checker_score = await self.arun_checker(
                    eval_result=transformed_result,
                    eval_score=transformed_score,
                    final_settings=final_settings,
                )

                return checker_result, checker_score

            return transformed_result, transformed_score

        # execute repetition
        if final_settings.repeat:
            # Parallel evaluation
            results_scores = await asyncio.gather(
                *(
                    asingle_evaluation() 
                    for _ in range(final_settings.repeat)
                )
            )
            results, scores = zip(*results_scores)
            results = tuple(results)
            scores = tuple(scores)
        else:
            results, scores = await asingle_evaluation()

        # apply aggregation
        aggregate_result, aggregate_score = await self.async_apply_aggregation(
            results, 
            scores, 
            final_settings
        )

        # check target on aggregate if aggregate defined
        if final_settings.aggregate:
            checker_result, checker_score = await self.arun_checker(
                eval_result=aggregate_result,
                eval_score=aggregate_score,
                final_settings=final_settings,
            )

            return checker_result, checker_score

        return aggregate_result, aggregate_score

    def sync_call(self, *call_args, **call_kwargs):
        
        final_settings, final_kwargs = self.get_final_settings_and_kwargs(call_kwargs)

        def single_evaluation() -> tuple[EvalResult, Any]:

            # run the evaluator
            score = self.func(*call_args, **final_kwargs)
            
            result = EvalResult(
                score=score, 
                init_method=self.name,
                eval_settings=final_settings,
                eval_kwargs=final_kwargs,
                func_impl=f"{self.func.__module__}.{self.func.__name__}",
                func_args=call_args,
                func_kwargs=call_kwargs,
                call_depth=call_context["depth"]
            )

            # transform
            transformed_result, transformed_score = self.sync_apply_transformation(
                result, 
                score, 
                final_settings
            )

            # check target on transform if aggregate not defined
            if not final_settings.aggregate:
                checker_result, checker_score = self.run_checker(
                    eval_result=transformed_result,
                    eval_score=transformed_score,
                    final_settings=final_settings,
                )

                return checker_result, checker_score

            return transformed_result, transformed_score

        # execute repetition
        if final_settings.repeat:
            # Serial evaluation
            results, scores = zip(
                *tuple(single_evaluation() for _ in range(final_settings.repeat))
            )
            results = tuple(results)
            scores = tuple(scores)
        else:
            results, scores = single_evaluation()

        # apply aggregation
        aggregate_result, aggregate_score = self.sync_apply_aggregation(
            results, scores, final_settings
        )

        # check target on aggregate if aggregate defined
        if final_settings.aggregate:
            checker_result, checker_score = self.run_checker(
                eval_result=aggregate_result,
                eval_score=aggregate_score,
                final_settings=final_settings,
            )

            return checker_result, checker_score

        return aggregate_result, aggregate_score

    def __call__(self, *args, **kwargs) -> Any:
        
        # RUN EVALUATOR
        try:
            call_context["depth"] += 1
            results, scores = None, None
            assert not asyncio.iscoroutinefunction(
                self.func
            ), "please use @aevaluator instead of @evaluator for this function"
            results, scores = self.sync_call(*args, **kwargs)
        finally:
            self.process_trace(results)
        return scores

    async def __acall__(self, *args, **kwargs) -> Any:
        
        # RUN EVALUATOR
        try:
            call_context["depth"] += 1
            results, scores = None, None
            if asyncio.iscoroutinefunction(self.func):
                results, scores = await self.async_call(*args, **kwargs)
            else:
                results, scores = self.sync_call(*args, **kwargs)
        finally:
            self.process_trace(results)
        return scores    

    def raw(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    async def araw(self, *args, **kwargs):
        return await self.func(*args, **kwargs)
    
    # ------------------------------------------------------------------------------
    # TRACING
    def process_trace(self, results):
        if results is None: return
        # if this is the first call, save the result.  Otherwise, add it to the list of previous runs
        if call_context["depth"] == 1:
            self._prev_run = results
        else:
            call_context["calls"].append(results)

        call_context["calls"].reverse()
        call_context["depth"] -= 1

        # TODO: this is only needed for recursion
        if call_context["depth"] < 0:
            if self._prev_run is not None:
                if isinstance(self._prev_run, (list, tuple)):
                    if len(self._prev_run) > 0:
                        # add the repeats to calls
                        for i, result in enumerate(self._prev_run):
                            if isinstance(result, EvalResult):
                                self._prev_run[i].metadata = {
                                    "prev_results": call_context["calls"]
                                }
                    else:
                        raise ValueError("Error tracing results: no results found!")
                else:
                    self._prev_run.metadata = {"prev_results": call_context["calls"]}
            else:
                print("Error tracing results!")
    # ------------------------------------------------------------------------------



    # ------------------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------------------

    @property
    def settings(self) -> EvalSettings:
        if self.name not in evaluator.all_evaluator_settings:
            evaluator.all_evaluator_settings[self.name] = EvaluatorSettings(name=self.name)
        return evaluator.all_evaluator_settings[self.name].resolve_settings()

    @settings.setter
    def settings(self, value: EvalSettings):
        assert isinstance(value, EvalSettings)
        if self.name not in evaluator.all_evaluator_settings:
            evaluator.all_evaluator_settings[self.name] = EvaluatorSettings(name=self.name)
        evaluator.all_evaluator_settings[self.name].explicit_settings = value

    @property
    def kwargs(self) -> dict[str, Any]:
        if self.name not in evaluator.all_evaluator_settings:
            evaluator.all_evaluator_settings[self.name] = EvaluatorSettings(name=self.name)
        return evaluator.all_evaluator_settings[self.name].resolve_kwargs()

    @kwargs.setter
    def kwargs(self, value: dict):
        assert isinstance(value, dict)
        if self.name not in evaluator.all_evaluator_settings:
            evaluator.all_evaluator_settings[self.name] = EvaluatorSettings(name=self.name)
        evaluator.all_evaluator_settings[self.name].explicit_kwargs = value

    @property
    def config(self) -> dict[str, Any]:
        if self.explicit_config is None:
            self.explicit_config = {
                **self.settings.dict(),
                **self.kwargs
            }
        return self.explicit_config
    
    @config.setter
    def config(self, value: Any):
        if value is None:
            self.explicit_config = None
        else:
            raise NotImplementedError

    @property
    def prev_run(self):
        if isinstance(self._prev_run, EvalResult):
            self._prev_run.trace()
        return self._prev_run

    @prev_run.setter
    def prev_run(self, value):
        call_context["calls"].append(self._prev_run)
        self._prev_run = value
    
    # ------------------------------------------------------------------------------
    # ACCESSORS
    # ------------------------------------------------------------------------------

    @classmethod
    def _validate_key(cls, key: str | Callable | None) -> str:
        if isinstance(key, str):
            return key
        elif isinstance(key, Callable):
            if hasattr(key, "__name__"):
                return key.__name__
            else:
                return str(key)
        else:
            raise KeyError(f"Invalid key type: {type(key)}")

    @classmethod
    def __class_getitem__(cls, keys):
        '''
        Get an evaluator by name or Callable.
        '''
        
        if isinstance(keys, (str, Callable)):
            # Ensure that key is a string
            key: str = cls._validate_key(keys)
            if key in cls.all_evaluators:
                return cls.all_evaluators[key]
            elif key in cls.all_evaluator_settings:
                # if the evaluator is wrapped, initialize and register the wrapper
                if (evaluator_settings := cls.all_evaluator_settings[key]) is not None:
                    # initialize and register the wrapper
                    evaluator.create_wrapped_evaluator(evaluator_settings)
                    return cls.all_evaluators[key]
            else:
                raise KeyError(f"Key '{key}' not found in evaluators or config.")
        elif isinstance(keys, tuple):
            # Multiple key access
            return [cls.__class_getitem__(key) for key in keys]
        else:
            raise KeyError(f"Invalid key type: {type(keys)}")

    @classmethod
    def __class_setitem__(cls, key, value):
        key = cls._validate_key(key)
        cls.all_evaluators[key] = value

    @classmethod
    def __class_delitem__(cls, key):
        key = cls._validate_key(key)
        del cls.all_evaluators[key]


class aevaluator(evaluator):

    async def __call__(self, *args, **kwargs):
        return await self.__acall__(*args, **kwargs)

    async def raw(self, *args, **kwargs):
        return await self.araw(*args, **kwargs)

