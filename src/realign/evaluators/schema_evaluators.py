from typing import Any, Union
import numpy as np

class EvalResult:
    def __init__(self, score: float = None, result: bool = None):
        self.score = score
        self.result = result

    def __bool__(self):
        return self.result
    
    def __repr__(self):
        return f'EvalResult(score={self.score}, result={self.result})'

class Checker:
    '''Check if a EvalResult's score is within a passing range and return the new EvalResult'''

    def __init__(self, 
                 method: str, 
                 *params, 
                 asserts: bool = False, 
                 target: Any = None):

        self.method = method
        self.params = params
        self.asserts = asserts
        self.target = target

    def check(self, eval_result: EvalResult, target = None) -> EvalResult:
        # if target is provided, use it
        target = target or self.target
        score = eval_result.score
        
        if self.method == 'cosine':
            '''Check if the cosine similarity score is within the threshold range'''
            threshold_min, threshold_max = self.params[0], self.params[1]
            cos_sim = self.cosine_similarity(eval_result, target)
            result = threshold_min <= cos_sim <= threshold_max

        elif self.method == 'contains_all':
            '''Check if all the targets contain the score'''

            assert type(target) in [list, tuple], 'target must be a list or tuple'
            assert hasattr(score, '__contains__'), 'score must have __contains__ method'
            
            result = all(t in score for t in target)

        elif self.method == 'contains_any_target':
            '''Check if any of the targets contain the score'''

            assert type(target) in [list, tuple], 'target must be a list or tuple'
            assert hasattr(score, '__contains__'), 'score must have __contains__ method'

            result = any(t in score for t in target)

        elif self.method == 'exact':
            result = score == target
        
        # if asserts is True, assert that the result is True
        if self.asserts:
            assert result

        # return a new EvalResult object with the computed result
        return EvalResult(eval_result.score, result)
    
    def __call__(self, eval_result, target = None) -> Any:
        return self.check(eval_result, target)

    @staticmethod
    def cosine_similarity(a, b):
        # Implement cosine similarity calculation
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class NumericRangeChecker(Checker):
    '''Evaluator for checking if a numeric value is within a passing range'''

    @staticmethod
    def in_num_interval(num_interval: Union[tuple, list], x):
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
    
    @staticmethod
    def is_equal(num, x):
        '''Checks if x is equal to num'''
        
        def check(x):
            return x == num

        # if score is iterable, check all elements
        if hasattr(x, '__iter__'):
            return all(check(x_i) for x_i in x)

        return check(x)

    @staticmethod
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

    def __init__(self, target: Any):
        
        in_interval = None
        if type(target) == str:
            in_interval = NumericRangeChecker.in_str_interval
        elif type(target) == list or type(target) == tuple:
            in_interval = NumericRangeChecker.in_num_interval
        elif type(target) in [int, float]:
            in_interval = NumericRangeChecker.is_equal
            
        super().__init__(target, in_range=in_interval)

class Eval:
    def __init__(self, 
                 eval_type: str, 
                 target: Any = None,
                 checker: str = None,
                 **stateless_settings):
        self.eval_type = eval_type
        self.target = target
        
        self.resolve_checker(checker)
        
        # save stateless settings
        self.statless_settings = stateless_settings
        
    def resolve_checker(self, checker: str = None):
        self.checker = Checker(checker, target=self.target) if checker else None

    def __call__(self, input_value: Any) -> EvalResult:
        if self.eval_type == 'contains_all':
            if not self.checker:
                self.resolve_checker('contains_all')

            score = input_value
            
        elif self.eval_type == 'tone':
            if not self.checker:
                self.resolve_checker('exact')

            score = "friendly" # mock

        if self.checker:
            return self.checker.check(EvalResult(score))

        return EvalResult(True, score)

    def __str__(self):
        return f"Eval(type='{self.eval_type}', target={self.target})"

# Usage examples
e = Eval(eval_type='contains_all', target=['h', 'e', 'l', 'o'])
print(e('hello'))

f = Eval('tone', provider='llm', target=['friendly', 'neutral', 'unfriendly'], checker='contains_any')
result = f('how are you')
print(result)

g = Eval('tone', target='friendly')
print(g('how are you'))