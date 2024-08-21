from lab.evaluators.evaluation import ChatEvaluation
from realign.datasets import ChatDataset
from lab.evaluators.llm_evaluators import allm_toxicity_rating

evaluation = ChatEvaluation()
evaluation.dataset = ChatDataset('data/run_data.json')

evaluation.evaluators = [ allm_toxicity_rating ]
evaluation.run()

evaluation.push_evals_dataset('data/eval_results.json')
