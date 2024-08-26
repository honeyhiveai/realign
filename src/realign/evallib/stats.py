import math

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


@evaluator
def elo_ratings(preferences: list[tuple[str, str]],
                initial_rating: int = 1500, 
                k_factor: int= 32
            ) -> list[tuple[str, int]]:
    
    '''
    preferences: list of tuples of (winner, loser)
    
    returns: list of tuples of (choice, rating)
    '''
    
    # Initialize ratings
    ratings = {
        choice: initial_rating
        for preference in preferences for choice in preference
    }
    
    # Process each preference
    for winner, loser in preferences:
        # Calculate expected scores
        exp_winner = 1 / (1 + math.pow(10, (ratings[loser] - ratings[winner]) / 400))
        exp_loser = 1 - exp_winner
        
        # Update ratings
        ratings[winner] += k_factor * (1 - exp_winner)
        ratings[loser] += k_factor * (0 - exp_loser)
    
    ranked_choices = sorted(ratings.items(), 
                            key=lambda x: x[1], 
                            reverse=True)
    
    return ranked_choices


