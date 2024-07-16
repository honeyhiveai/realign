from agentsim import Simulator
# from agentsim.utils import print_system_prompt, print_chat, evaluations_dataframe
from dotenv import load_dotenv


def main():
    load_dotenv()

    sim = Simulator('tests/config.yaml')
    print_system_prompt(sim, app_agents=['teacher_agent'])
    
    messages = []
    while len(messages) < 30:
        sim.sim_agent.process_turn(messages)
        sim.app_agent.process_turn(messages)
        print_chat(messages[-2:])
    
    evaluation = sim.evaluators['toxicity_score_llm'](messages)
    if evaluation.result:
        print('Toxicity score is within the acceptable range')

    print(toxicity)

if __name__ == '__main__':
    main()
