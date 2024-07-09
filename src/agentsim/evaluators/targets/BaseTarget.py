from typing import Optional, Any, Callable

# a callable evaluator class
# has 2 components: a pass_range, and an in_range function
class BaseTarget(object):
    
    DEFAULT_PASS_RANGE: Optional[Any] = True
    DEFAULT_IN_RANGE: Callable[[Any, Any], bool] \
        = lambda pass_range, score: score == BaseTarget.DEFAULT_PASS_RANGE

    def __init__(self, 
                 pass_range: Optional[Any] = None,
                 in_range: Callable[[Any, Any], bool] = None):

        self._pass_range = pass_range or BaseTarget.DEFAULT_PASS_RANGE

        if in_range is not None:
            self._in_range = in_range
        elif pass_range is not None:
            self._in_range = lambda pass_range, score: score == pass_range
        else:
            self._in_range = BaseTarget.DEFAULT_IN_RANGE
    
    # calling the evaluator runs the in_range function on pass_range
    def __call__(self, *in_range_args, **in_range_kwargs):
        return self._in_range(self._pass_range, *in_range_args, **in_range_kwargs)
    
    # run the eval over a custom range with the same in_range function
    def with_custom_pass_range(self, custom_pass_range, *in_range_args, **in_range_kwargs):
        assert custom_pass_range is not None, 'custom_pass_range must be provided'
        return BaseTarget(custom_pass_range, self._in_range, *in_range_args, **in_range_kwargs)

    # run the eval over the same range with a custom in_range function
    def with_custom_in_range(self, custom_in_range, *in_range_args, **in_range_kwargs):
        assert custom_in_range is not None, 'custom_in_range must be provided'
        return BaseTarget(self._pass_range, custom_in_range, *in_range_args, **in_range_kwargs)

    # check if a score is in the pass_range
    # example: 0.8 in eval -> True
    def __contains__(self, score):
        return self._in_range(self._pass_range, score)

    def __repr__(self) -> str:
        return f'{self._pass_range}'
