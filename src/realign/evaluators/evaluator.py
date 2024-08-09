from typing import Any, Callable, Optional
from realign.types import EvalResult

# score 

# optional:
# in range
# target



class Eval:
    '''A callable class which runs an evaluator. Initialized with a string of the evaluator name'''

    def __init__(
        self,
        type: str,

        checker: Optional[str] = None,
        target: Optional[Any] = None,
        
        weight: Optional[float] = 1.0,

        **settings: Optional[Any],
    ) -> None:
        
        # evaluator type
        self.type = type
        
        # checks if the score is in range of the target
        self.checker = checker
        self.target = target
        
        self.weight = weight
        # self.eval_func: Callable[..., EvalResult] = lambda x: x

        self.settings = settings
        
        return self.resolve_evaluator()
        
    def resolve_evaluator(self):
        if self.type == 'contains_all':
            return ContainsAll(self.type, self.checker, self.target, self.weight, **self.settings)

    async def acall(self, *states: Any, **settings):
        return await self.eval_func(*states, **settings)

    async def __call__(self, *states, **settings):
        return await self.eval_func(*states, **settings)
    
    def __call__(self, *states, **settings):
        return self.eval_func(*states, **settings)
    
    def check(self, *states, **settings):
        score = self.eval_func(*states, **settings)
        return self.checker(score)

    def eval_func(self, *states, **settings):
        return states
    
    def checker(self, score):
        return score == (self.target if self.target is not None else True)

class ContainsAll(Eval):
    '''Measures whether all elements in a list are contained in another list'''
    def __init__(self, *eval_args, **settings):
        super().__init__(*eval_args, **settings)
        if self.eval_func is None:
            assert type(self.target) == list
            
            self.eval_func = lambda score: all([t in score for t in self.target])

    def eval_func(self, *states, **settings):
        if len(states) == 0:
            return False
        assert hasattr(states[0], '__contains__')
        if len(self.target) == 0:
            target = [target]
        else:
            target = self.target
        return [t in states[0] for t in target]

    def checker(self, score):
        return all(score)


e = Eval(type='contains_all', target=['h', 'e', 'l', 'o'])
print(e)
print(e('hello'))

f = Eval('tone', provider='llm', target='hello', checker='cosine-[0.8,1]-assert')

f('how are you').score # score
f('how are you').result # evaluation result True/False


# YAML



# name of the checker
# 


class Custom(Eval):
    
    def __init__(self, type: str, checker: str | None = None, target: Any | None = None, weight: float | None = 1, **settings: Any | None) -> None:
        self.type = 'custom'
        super().__init__(type, checker, target, weight, **settings)
        
    def eval_func(self, state, **stateless):
        return None

f('hello').score == 'hello'
f('hello').guardrail

