# this object contains a single score
# the score can be of any type, including a list of scores
class Evaluation(object):
    def __init__(self, score, result, state=None, eval_name=None):
        self.score = score
        self.result = result
        self.state = state
        self.eval_name = eval_name

    def __repr__(self):
        return f'(eval_name: {self.eval_name}, score: {self.score}, result: {self.result})'
    
    def unpack(self):
        return self.score, self.result
    
    def __dict__(self):
        return {
            'eval_name': self.eval_name,
            'score': self.score,
            'result': self.result
        }
    
    def to_dict(self):
        return self.__dict__()

def validate_evaluator_response_schema(evaluator_response):

    # verify that the evaluator response is a tuple of 2 elements (score: Any, result: bool | None)
    if type(evaluator_response) not in [list, tuple] or \
        len(evaluator_response) != 2 or \
            type(evaluator_response[1]) not in [bool, type(None)]:
        raise ValueError('Evaluator response must be a tuple of 2 elements, the score of type Any and result of type bool | None')

# evaluator decorator to wrap an app scorer inside a Score object
def evaluator(evaluator) -> Evaluation:
    def wrapper(state):
        import asyncio
        
        # Check if the score function is async
        if asyncio.iscoroutinefunction(evaluator):
            # If evaluator is async, return a Future Score object
            async def async_wrapper():
                # call the evaluator
                response = await evaluator(state)
                
                # validate the response schema
                validate_evaluator_response_schema(response)

                # unpack results
                score, result = response
                return Evaluation(score, result, state, evaluator.__name__)
            return async_wrapper()
        else:
            # call the evaluator
            response = evaluator(state)

            # validate the response schema
            validate_evaluator_response_schema(response)

            # unpack results
            score, result = response
            return Evaluation(score, result, state, evaluator.__name__)
    return wrapper

