
from realign.evaluators import aevaluator, evaluator
from realign.llm_utils import allm_messages_call


@aevaluator
async def llm_rating_json(criteria=None, 
                          messages=None,
                          agent_settings=None):
    
    message = await allm_messages_call(
        template='rating_5_star',
        template_params={
            'messages': messages,
            'criteria': criteria
        },
        agent_settings=agent_settings
    )
    return message.content

@aevaluator
async def llm_rating_json_aggregate(values: tuple[dict[str, str]]):
    # rating_5_star returns a json with an explanation and a rating
    
    # derive the mean rating
    # derive the explanation summary
    
    mean_rating = sum(float(v['rating']) for v in values) / len(values)
    
    
    explanations = ''
    for i, v in enumerate(values):
        explanations += f'Explanation {i+1}: {v["explanation"]}\n'
        
    explanation_summary = await allm_messages_call(
        template='explanation_summary',
        template_params={
            'explanations': explanations
        }
    )
    
    return {
        'rating': mean_rating,
        'explanation_summary': explanation_summary.content
    }
