from realign.evaluators import evaluator

@evaluator
def numrange(x, target=None):
    '''Evaluator for checking if a numeric value is within a passing range'''
    
    assert x, 'numrange requires an argument'
    
    if not target:
        return True
    
    num_range = target
    
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
