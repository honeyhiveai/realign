from realign.types import EvalSettings, EvalResult
from realign.config import get_eval_settings, extract_eval_settings_and_kwargs
from realign.evallib import resolve_evaluator_type, get_python_globals

from typing import Optional, Any, Callable
import asyncio

class Eval:
    
    def load_evaluate(self, type_or_func: Optional[str | Callable] = None) -> None:
        # load the evaluate function
        assert type_or_func is not None, 'Please provide an eval_type or a function'
        
        if isinstance(type_or_func, str):
            self.eval_type = type_or_func
            self.evaluate = resolve_evaluator_type(self.eval_type)
        elif isinstance(type_or_func, Callable):
            self.eval_type = type_or_func.__name__
            self.evaluate = type_or_func

        assert self.evaluate is not None, f'No evaluator of type {self.eval_type} found'     
        
    def load_checker(self) -> None:
        if self.settings.checker:
        
            # load the checker function
            checker_str_or_func = self.settings.checker
            
            if isinstance(checker_str_or_func, str):
                self.checker = resolve_evaluator_type(checker_str_or_func)
            elif isinstance(checker_str_or_func, Callable):
                self.checker = checker_str_or_func

            assert self.checker is not None, f'No evaluator of type {self.eval_type} found'

    def load_globals(self, inject_globals: Optional[dict] = None) -> None:
        
        # basic python builtin functions
        self._evallib_globals: dict = get_python_globals()
        
        # load methods from config
        self._evallib_globals.update(resolve_evaluator_type())
        
        # inject user given globals
        if inject_globals:
            self._evallib_globals.update(inject_globals)
        
    def apply_transformation(self, eval_result: EvalResult) -> EvalResult:
        
        transform_expr = self.settings.transform
        if not transform_expr:
            return eval_result
        
        globals_dict = {
            'result': eval_result,
            **self._evallib_globals
        }
        
        raw_result = eval(transform_expr, globals_dict)
        
        transformed_result = EvalResult(raw_result, prev_result=eval_result)
        if type(raw_result) == EvalResult:
            transformed_result.init_method = raw_result.init_method
        
        return transformed_result
    
    def apply_aggregation(self, eval_result_tuple: tuple[EvalResult] | EvalResult) -> EvalResult:
        
        aggregation_expr = self.settings.aggregate
        if not aggregation_expr:
            return eval_result_tuple
        
        globals_dict = {
            **self._evallib_globals
        }

        if type(eval_result_tuple) == EvalResult:
            globals_dict['results'] = eval_result_tuple.score
        else:
            globals_dict['results'] = eval_result_tuple
        
        raw_result = eval(aggregation_expr, globals_dict)
        
        init_method = None
        if type(raw_result) == EvalResult:
            init_method = raw_result.init_method
        
        aggregated_result = EvalResult(raw_result,
                                       init_method=init_method,
                                       prev_results=eval_result_tuple)
        return aggregated_result
    
    def __init__(self, 
                 type_or_func: Optional[str | Callable] = None,
                 inject_globals: Optional[dict] = None,
                 **setting_kwargs) -> None:
        
        '''Things that are passed by reference need to be pulled up'''
        
        # evaluate function
        self.evaluate: Callable = None
        
        # checker function
        self.checker: Callable = None
        
        # evaluator settings
        self.settings: EvalSettings = None
        
        # evaluator kwargs
        self.eval_kwargs: dict = None
        
        # evaluator type
        self.eval_type: str = None
        
        # load the evaluate function
        self.load_evaluate(type_or_func)

        # get the settings in the config file
        self.settings, self.eval_kwargs = get_eval_settings(eval_type=self.eval_type)
        
        # override settings with setting_kwargs if provided
        self.settings, self.eval_kwargs = self.get_settings(**setting_kwargs)
                
        # load the checker function
        self.load_checker()
        
        # load the evallib globals
        self.load_globals(inject_globals)
        
    def get_settings(self, **setting_kwargs) -> tuple[EvalSettings, dict]:
        """
        Returns EvalSettings.
        If type is provided, get the settings for that type. 
        If setting_kwargs is provided, update the settings with the provided values.
        
        Always returns a copy without modifying the original settings.
        """
        
        if setting_kwargs:
        
            extracted_settings, extracted_kwargs = extract_eval_settings_and_kwargs(setting_kwargs)

            # create a copy of the settings and eval_kwargs
            settings = self.settings.copy()
            eval_kwargs = self.eval_kwargs.copy()

            # update the settings with the provided values
            if extracted_settings:
                for key, value in extracted_settings.items():
                    setattr(settings, key, value)
            
            if extracted_kwargs:
                for key, value in extracted_kwargs.items():
                    eval_kwargs[key] = value
            
            return settings, eval_kwargs
        
        return self.settings, self.eval_kwargs
    
    
    def run_checker(self, eval_result: EvalResult, target=None, asserts=None) -> bool:
        target = str(target) if target else True
        if self.checker:
            checker_result = self.checker(eval_result, target=target)
            if asserts:
                assert checker_result, f'Assertion failed: score {eval_result} is not in range {target}'
            eval_result = checker_result
        return eval_result
        
                
    @staticmethod
    def call(*x, 
             type_or_func: Optional[str | Callable] = None, 
             **setting_kwargs) -> tuple[EvalResult]:
        
        e = Eval(type_or_func, **setting_kwargs)
        
        return e.call(*x)
    
    @staticmethod
    async def acall(*x,
                type_or_func: Optional[str | Callable] = None,
                **setting_kwargs) -> tuple[EvalResult]:
        
        e = Eval(type_or_func, **setting_kwargs)
        
        return await e.acall(*x)
        
    async def acall(self,
                    *x,
                    **setting_kwargs) -> tuple[EvalResult] | EvalResult:
        
        assert asyncio.iscoroutinefunction(self.evaluate), 'evaluate must be an async function when using acall'
        
        settings, eval_kwargs = self.get_settings(**setting_kwargs)
        
        async def single_evaluation():
            # evaluate
            eval_result = await self.evaluate(*x, **eval_kwargs)
            if not isinstance(eval_result, EvalResult):
                eval_result = EvalResult(eval_result)
                
            assert isinstance(eval_result, EvalResult), 'eval result must be of type EvalResult'
                
            # apply the actual settings
            eval_result.eval_settings = settings
            eval_result.eval_kwargs = eval_kwargs
            eval_result.weight = settings.weight        
            
            # transform
            eval_result = self.apply_transformation(eval_result)
            
            # check target on transform if aggregate not defined
            if not self.settings.aggregate:
                eval_result = self.run_checker(eval_result, 
                                 target=settings.target,
                                 asserts=settings.asserts)

            return eval_result
        
        if settings.repeat:
            # Parallel evaluation
            results = await asyncio.gather(
                *(
                    single_evaluation() 
                    for _ in range(settings.repeat)
                )
            )
            results = tuple(results)
        else:
            results = await single_evaluation()
        
        final_result = self.apply_aggregation(results)
        
        # check target on aggregate if aggregate defined
        if self.settings.aggregate:
            final_result = self.run_checker(final_result, 
                                 target=settings.target,
                                 asserts=settings.asserts)
        
        return final_result
    
    def call(self, 
             *x: Any, 
             **setting_kwargs) -> tuple[EvalResult] | EvalResult:
        
        assert not asyncio.iscoroutinefunction(self.evaluate), 'evaluate must be a sync function when using call'
        
        settings, eval_kwargs = self.get_settings(**setting_kwargs)
        
        def single_evaluation():
            # evaluate
            eval_result = self.evaluate(*x, **eval_kwargs)
            if not isinstance(eval_result, EvalResult):
                eval_result = EvalResult(eval_result)
                
            assert isinstance(eval_result, EvalResult), 'eval result must be of type EvalResult'
                
            # apply the actual settings
            eval_result.eval_settings = settings
            eval_result.eval_kwargs = eval_kwargs
            eval_result.weight = settings.weight
            
            # transform
            eval_result = self.apply_transformation(eval_result)
            
            # check target on transform if aggregate not defined
            if not self.settings.aggregate:
                eval_result = self.run_checker(eval_result, 
                                 target=settings.target,
                                 asserts=settings.asserts)

            return eval_result
        
        if settings.repeat:
            # Serial evaluation
            results = [
                single_evaluation() 
                for _ in range(settings.repeat)
            ]
            results = tuple(results)
        else:
            results = single_evaluation()
            
        final_result = self.apply_aggregation(results)
        
        # check target on aggregate if aggregate defined
        if self.settings.aggregate:
            final_result = self.run_checker(final_result, 
                                 target=settings.target,
                                 asserts=settings.asserts)
        return final_result
    