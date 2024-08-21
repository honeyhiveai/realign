
from realign.agents import ChatAgent, SyntheticUserFactory
from realign.simulation import ChatSimulation
from lab.evaluators.llm_evaluators import allm_summary

# initialize simulation
simulation = ChatSimulation(runs=2, max_messages=6)

# to ignore rate limiting, uncomment the following line
simulation.router_settings = {
    '*/*': '*',
}

# initialize your app agent
simulation.app = ChatAgent(system_prompt='''
    As an AI tutor, your role is to guide student learning across various subjects through explanations and questions. \
    Assess student knowledge and adapt your approach accordingly, providing clear explanations with simple terms and examples. \
    Keep responses very concise (maximum 1 sentence) and to the point, and avoid jargon or complex terms. \
''', model='openai/gpt-4o-mini')

# initialize your synthetic user agent builder
simulation.simulator = SyntheticUserFactory().as_a('someone who wants to learn something new').they_want_to('learn a new complex subject')

# to use a different model for the synthetic user agent
# simulation.simulator.with_synth_user_model('openai/gpt-4o-mini')

# add evaluators
simulation.evaluators = [allm_summary]

# run simulation
simulation.run()

# publish simulation run and eval results
simulation.push_runs_to_dataset('data/run_data.json')
simulation.push_evals_dataset('data/eval_data.json')
