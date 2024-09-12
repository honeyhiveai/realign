import numpy as np
import itertools

from realign.evaluators import aevaluator, evaluator
from realign.llm_utils import allm_messages_call, llm_messages_call, AgentSettings, all_agent_settings
from realign.utils import run_async


@aevaluator
async def allm_rating_json(messages=None, 
                          criteria=None, 
                          agent_settings=None):
    """
    (async) Use an LLM to get a rating for a text sample.

    Args:
        messages (list[OpenAIMessage], optional): The messages to be rated.
        criteria (str, optional): The criteria for rating the messages.
        agent_settings (AgentSettings, optional): Settings for the LLM agent.

    Returns:
        float: The rating given by the LLM.

    Note:
        This function uses an asynchronous LLM call to generate a rating based on the provided messages and criteria.
        The rating is expected to be in a 5-star format.
    """

    message = await allm_messages_call(
        template="rating_5_star",
        template_params={"messages": messages, "criteria": criteria},
        agent_settings=agent_settings,
    )
    
    return message.content

@evaluator
def llm_rating_json(messages=None, criteria=None, agent_settings=None):
    message = llm_messages_call(
        template="rating_5_star",
        template_params={"messages": messages, "criteria": criteria},
        agent_settings=agent_settings,
    )
    return message.content

@aevaluator
async def allm_pairwise_judge(
                       choices,
                       batch_size=2,
                       criteria=None,
                       agent_settings=None,
                    ) -> tuple[str, str]:
    """
    Perform pairwise judgments on a list of choices using an LLM.

    This function compares choices in batches, determining the best and worst options
    within each batch according to specified criteria.

    Args:
        choices (list): A list of choices to be compared.
        batch_size (int, optional): The number of choices to compare in each batch. Defaults to 2.
        criteria (str, optional): The criteria for judging the choices. Defaults to None.
        agent_settings (AgentSettings, optional): Settings for the LLM agent. Defaults to None.

    Returns:
        tuple[str, str]: A tuple containing the best and worst choices overall.

    Raises:
        AssertionError: If the batch size is greater than the number of choices.

    Note:
        This function uses an asynchronous LLM call to generate comparisons based on the provided choices and criteria.
        It processes all possible permutations of the choices in batches for comprehensive comparison.
    """
    
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
async def allm_rating_json_aggregate(values: tuple[dict[str, str]]):
    """
    Aggregates multiple JSON ratings and their explanations.

    This asynchronous function takes a tuple of dictionaries, each containing a rating and an explanation,
    and aggregates them into a single summary. It calculates the mean and variance of the ratings,
    concatenates all explanations, and generates a summary of the explanations using an AI language model.

    Args:
        values (tuple[dict[str, str]]): A tuple of dictionaries, where each dictionary contains
                                        'rating' and 'explanation' keys.

    Returns:
        dict: A dictionary containing:
            - rating_mean (float): The mean of all ratings.
            - rating_variance (float): The variance of all ratings.
            - explanation_summary (str): A summary of all explanations generated by an AI model.
            - raw_ratings (list): A list of all individual ratings.

    Note:
        This function requires the `allm_messages_call` function to be available in the scope
        for generating the explanation summary.
    """

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

@aevaluator
async def allm_judge(agent_name: str = None, **kwargs):
    """
    Judge a set of values using an LLM.
    """
    message = await allm_messages_call(
        agent_name=agent_name,
        **kwargs
    )
    return message.content