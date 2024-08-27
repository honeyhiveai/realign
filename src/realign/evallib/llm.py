import numpy as np
import itertools

from realign.evaluators import aevaluator, evaluator
from realign.llm_utils import allm_messages_call, AgentSettings
from realign.utils import run_async


@aevaluator
async def llm_rating_json(messages=None, 
                          criteria=None, 
                          agent_settings=None):

    message = await allm_messages_call(
        template="rating_5_star",
        template_params={"messages": messages, "criteria": criteria},
        agent_settings=agent_settings,
    )
    
    return message.content

@aevaluator
async def allm_choice_judge(
                       choices,
                       batch_size=2,
                       criteria=None,
                       agent_settings=None,
                    ) -> tuple[str, str]:
    
    assert batch_size <= len(choices), "Batch size must be less than or equal to the number of choices"
    
    # all batches of non-similar choices
    comparison_idxs = list(itertools.permutations(list(range(len(choices))), batch_size))
    
    print(f'Running Np2 = {len(choices)}p{batch_size} = {len(comparison_idxs)} comparisons concurrently')
    print(comparison_idxs)
    
    comparison_tasks = []
    
    # run the comparisons
    for batch_idxs in comparison_idxs:
        
        async def batch_compare_task():
            
            batch_choices = [str(choices[i]) for i in batch_idxs]
            
            message = await allm_messages_call(
                template_params={
                    "choices": batch_choices,
                    "criteria": criteria
                },
                agent_settings=agent_settings,
            )
            
            best_choice = int(message.content['best_choice']) - 1
            worst_choice = int(message.content['worst_choice']) - 1
            
            assert 0 <= best_choice < len(choices), f"Best choice {best_choice} out of range"
            assert 0 <= worst_choice < len(choices), f"Worst choice {worst_choice} out of range"
            
            # Generate pairwise comparisons
            all_preferences = [
                (choices[best_choice], choices[worst_choice])
            ]
            for idx in batch_idxs:
                if idx != best_choice and idx != worst_choice:
                    pair = (choices[best_choice], choices[idx])
                if idx != worst_choice and idx != best_choice:
                    pair = (choices[idx], choices[worst_choice])

                all_preferences.append(pair)
            return all_preferences
        
        comparison_tasks.append(batch_compare_task())
    
    all_batch_results = await run_async(comparison_tasks)
    print('Ran', len(all_batch_results), 'comparisons concurrently')
    
    # flatten the rankings
    pairs = [
        pair 
        for batch in all_batch_results 
        for pair in batch
    ]
    
    # pairs are a list of (winner, loser)
    return pairs

@aevaluator
async def llm_rating_json_aggregate(values: tuple[dict[str, str]]):
    # rating_5_star returns a json with an explanation and a rating

    # derive the mean rating
    # derive the explanation summary

    ratings = np.array([float(v["rating"]) for v in values])
    rating_mean = np.mean(ratings)
    rating_variance = np.var(ratings)

    explanations = ""
    for i, v in enumerate(values):
        explanations += f'Explanation {i+1}: {v["explanation"]}\n'

    explanation_summary = await allm_messages_call(
        template="explanation_summary",
        template_params={"explanations": explanations}
    )

    return {
        "rating_mean": float(rating_mean), 
        "rating_variance": float(rating_variance),
        "explanation_summary": str(explanation_summary.content),
        "raw_ratings": list(ratings),
    }
