from realign.evaluation import ChatEvaluation
from realign.datasets import ChatDataset
from realign.evaluators.llm_evaluators import allm_general_score

evaluation = ChatEvaluation()
evaluation.dataset = ChatDataset('src/realign/data/run_data.json')

evaluation.evaluators = [ allm_general_score ]
evaluation.run().cluster_evals()


# evaluation.push_evals_dataset('src/realign/data/eval_results.json')

# evaluation.show_results()

# 

# inputs
# outputs
# ground_truths
# metadata

# Evaluation
# generate inputs -> seed.build() -> seed.simulator.process_turn() 
# generate outputs -> seed.app.process_turn()
# run evaluators



dataset = ChatDataset('src/realign/data/run_data.json')
simulator = SyntheticUserBuilder().as_a('photographer').they_want_to('learn technical aspects of photography')

# seed = seed

dataset.generate_inputs(input_features=['topic', 'tone'], app_description=app_system_prompt)



dataset.simulate(simulator)