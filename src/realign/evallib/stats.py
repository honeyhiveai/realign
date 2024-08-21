from realign.evaluators import evaluator, EvalResult

@evaluator
def weighted_sum(values, results: list[EvalResult]):
    
    weighted_sum = 0
    for value, result in zip(values, results):
        assert hasattr(result, 'weight'), 'All results must have a weight'
        weighted_sum += float(value) * result.weight
    
    return weighted_sum

@evaluator
def weighted_mean(values, results: list[EvalResult]):

    return weighted_sum(values, results) / sum(result.weight for result in results)

