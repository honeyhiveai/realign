
from realign.types import ModelSettings
from realign.evaluation import evaluator
from realign.agents import ChatAgent, SyntheticUserBuilder
from realign.simulation import ChatSimulation
from realign.llm_utils import allm_messages_call


# unit test
if __name__ == '__main__':
    
    import numpy as np
    
    @evaluator
    def aggregate_length(messages):

        def length_evaluator(messages):
            print('Length evaluator', len(messages))
            return len(messages), True
        
        evaluations = [length_evaluator(messages) for _ in range(3)]
        
        # unzip into scores and results
        scores, results = zip(*evaluations)

        return np.mean(scores), all(results)
        
    
    @evaluator(repeat=3)
    async def llm_debate_winner(messages):
        
        model_settings_base = ModelSettings(
            model='openai/gpt-4o',
            role='assistant',
            template='rating_5_star',
            json_mode=True,
        )
        trump_settings = model_settings_base
        model_settings_base.prompt_params = {'criteria': 'Rate Trump\'s performance vs Biden in this debate.'}
        response_trump = await allm_messages_call(model_settings=trump_settings, messages=messages)
        
        biden_settings = model_settings_base
        model_settings_base.prompt_params = {'criteria': 'Rate Biden\'s performance vs Trump in this debate.'}
        
        response_biden = await allm_messages_call(model_settings=biden_settings, messages=messages)
        
        trump_score_vs_biden = response_trump.content['rating'] - response_biden.content['rating']
        print('Trump score vs Biden:', trump_score_vs_biden)
        return trump_score_vs_biden, trump_score_vs_biden >= 0
    
    @evaluator
    def user_role_counter(messages):
        user_messages = [m for m in messages if m.role == 'user']
        print('user role counter', len(user_messages))
        return len(user_messages), True
    

    # build a synthetic user
    synth_user_builder = SyntheticUserBuilder().as_a('photographer').they_want_to('learn technical aspects of photography')
    
    simulation = ChatSimulation(runs=3, max_messages=5)

    simulation.app = ChatAgent(system_prompt='''
As an AI tutor, your role is to guide student learning across various subjects through explanations and questions. Assess student knowledge and adapt your approach accordingly, providing clear explanations with simple terms and examples. Encourage critical thinking, offer step-by-step problem-solving guidance, and give constructive feedback. Be flexible in addressing different learning styles while maintaining a friendly, encouraging tone. Focus on academic subjects, promote understanding over mere answer-giving, and admit knowledge limitations when necessary. Ensure safe, appropriate interactions and tailor your language to the student's age and level. Your goal is to support learning without replacing human teachers or doing the student's work for them.
        ''',
        model='groq/llama3-8b-8192', role='assistant')
    
    simulation.simulator = synth_user_builder

    simulation.evaluators = [aggregate_length, llm_debate_winner]

    # simulation.dataset = ChatDataset('src/realign/data.json')
    # print(simulation.dataset.data)
    
    # add to new dataset
    # continue trajectory
    # continue dataset
    # regenerate a similar dataset
    
    # evaluation.evaluators = [length_evaluator, user_role_counter, llm_debate_winner]
    # simulation.evaluators = [length_evaluator, user_role_counter]

    simulation.run()
    
    # publish run and eval results
    simulation.push_runs_to_dataset('src/realign/run_data.json')
    simulation.push_evals_dataset('src/realign/eval_data.json')
    