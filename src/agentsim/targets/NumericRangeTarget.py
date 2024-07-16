from typing import Any, Union
from agentsim.targets.BaseTarget import BaseTarget

class NumericRangeTarget(BaseTarget):
    '''Evaluator for checking if a numeric value is within a passing range'''

    @staticmethod
    def in_num_interval(num_interval: Union[tuple, list], x):
        '''Checks if x is between two numbers in num_interval, inclusive of the bounds'''
        
        assert type(num_interval) in [tuple, list] and len(num_interval) == 2, \
            'pass_range must be a tuple or list of length 2'

        left, right = num_interval
        
        def check(x):
            return float(left) <= x <= float(right)

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

    def __init__(self, pass_range: Any):
        
        in_interval = None
        if type(pass_range) == str:
            in_interval = NumericRangeTarget.in_str_interval
        elif type(pass_range) == list or type(pass_range) == tuple:
            in_interval = NumericRangeTarget.in_num_interval
        elif type(pass_range) in [int, float]:
            in_interval = NumericRangeTarget.is_equal
            
        super().__init__(pass_range, in_range=in_interval)
