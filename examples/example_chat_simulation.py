
from realign.agents import ChatAgent, SyntheticUserBuilder
from realign.simulation import ChatSimulation
from realign.evaluators.llm_evaluators import allm_toxicity_rating, allm_user_engagement

# initialize simulation
simulation = ChatSimulation(runs=3, max_messages=2)

# to ignore rate limiting, uncomment the following line
# simulation.router_settings = {
#     '*/*': '*',
# }

# initialize your app agent
simulation.app = ChatAgent(system_prompt='''
    As an AI tutor, your role is to guide student learning across various subjects through explanations and questions. \
    Assess student knowledge and adapt your approach accordingly, providing clear explanations with simple terms and examples. \
    Keep responses very concise (maximum 1 sentence) and to the point, and avoid jargon or complex terms. \
''', model='openai/gpt-4o-mini')

# initialize your synthetic user agent builder
simulation.simulator = SyntheticUserBuilder().as_a('undergrad student').they_want_to('learn something new')

# to use a different model for the synthetic user agent
# simulation.simulator.with_synth_user_model('openai/gpt-4o-mini')

# add evaluators
simulation.evaluators = [allm_toxicity_rating, allm_user_engagement]

# run simulation
simulation.run()

# publish simulation run and eval results
simulation.push_runs_to_dataset('data/run_data.json')
simulation.push_evals_dataset('data/eval_data.json')
