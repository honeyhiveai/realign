
from realign.llm_utils import evaluator
from realign.agents import ChatAgent, SyntheticUserBuilder
from realign.simulation import ChatSimulation
from realign.datasets import ChatDataset
from realign.evaluators.llm_evaluators import allm_toxicity_rating
import numpy as np

# unit test
if __name__ == '__main__':
    
    simulation = ChatSimulation(runs=5, max_messages=3)

    simulation.app = ChatAgent(
        system_prompt='''
            As an AI tutor, your role is to guide student learning across various subjects through explanations and questions. Assess student knowledge and adapt your approach accordingly, providing clear explanations with simple terms and examples. Encourage critical thinking, offer step-by-step problem-solving guidance, and give constructive feedback. Be flexible in addressing different learning styles while maintaining a friendly, encouraging tone. Focus on academic subjects, promote understanding over mere answer-giving, and admit knowledge limitations when necessary. Ensure safe, appropriate interactions and tailor your language to the student's age and level. Your goal is to support learning without replacing human teachers or doing the student's work for them.
        ''',
        model='groq/llama3-8b-8192', 
        role='assistant')

    # 
    simulation.simulator = SyntheticUserBuilder().as_a('photographer') \
                                                 .they_want_to('learn technical aspects of photography')
    
    # simulation.dataset = ChatDataset('path').generate()
    
    @evaluator
    def aggregate_length(messages):

        def length_evaluator(messages):
            print('Length evaluator', len(messages))
            return len(messages), True
        
        evaluations = [length_evaluator(messages) for _ in range(3)]
        
        # unzip into scores and results
        scores, results = zip(*evaluations)

        return np.mean(scores), all(results)
    
    @evaluator
    def user_role_counter(messages):
        user_messages = [m for m in messages if m.role == 'user']
        print('user role counter', len(user_messages))
        return len(user_messages), True

    simulation.evaluators = [ allm_toxicity_rating ]

    # simulation.run()
    
    # publish run and eval results
    # simulation.push_runs_to_dataset('src/realign/data/run_data.json')
    # simulation.push_evals_dataset('src/realign/data/eval_data.json')


    # add to new dataset
    # continue trajectory
    # continue dataset
    # regenerate a similar dataset
    
    dataset = ChatDataset('src/realign/data/run_data.json') \
                    .with_seed(persona='photographer', scenario='get help on technical aspects of photography') \
                    .generate_inputs(input_template='I would like help on {{task}}', num_inputs=3)
    
    print(dataset.data['inputs'])
                    
                    
    # synth user
    # model settings
        # model
        # system prompt / template + params
        # hyperparams
        # json mode
                    
    # single turn. run_app is a 1 turn simulation
    dataset = ChatDataset().with_seed(persona, intent).generate_inputs().run_app(app).evaluate()
    
    # # multi turn. simulate is a multi turn simulation 
    dataset = ChatDataset().with_seed(persona, intent).generate_inputs().simulate(app, turns=10).evaluate()
    
    
    # seed = dataset.get_seed()
    
    # new_dataset = ChatDataset().with_seed(seed).generate_inputs().simulate(app, turns=10).evaluate()
    # new_dataset.export_results('path')
    
    # # continue lmsys chats for 10 turns. Reverse engineer the seed
    # dataset = ChatDataset(source='hf/lmsys').simulate(app, turns=10).evaluate()
    
    # # seed is the data used to generate inputs
    # lmsys_inputs = LMSysDataset().derive_inputs()
    # dataset = ChatDataset().with_inputs(lmsys_inputs).simulate(app, turns=10).evaluate()
    
    # dataset = LMSysDataset().evaluate()
    
    
    # # dataset
    # # seed
    # # inputs
    # # outputs
    
    # simulation = ChatSimulation(runs=5, max_messages=3)
    # # simulation.dataset = ChatDataset('path').generate()

    
    # customer support agent: connects to docs
    # RAG: content production
    # coding agent