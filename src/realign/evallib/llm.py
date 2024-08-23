import numpy as np

from realign.evaluators import aevaluator, evaluator
from realign.llm_utils import allm_messages_call


@aevaluator
async def llm_rating_json(messages=None, criteria=None, agent_settings=None):

    message = await allm_messages_call(
        template="rating_5_star",
        template_params={"messages": messages, "criteria": criteria},
        agent_settings=agent_settings,
    )
    
    return message.content


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
