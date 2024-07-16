from agentsim import Simulator
from agentsim.evaluators import message_limit, toxicity_score_llm, check_introduction_assessment, router, check_concept_explanation
from agentsim.utils import print_system_prompt, print_chat, evaluations_dataframe
from dotenv import load_dotenv


def main():
    load_dotenv()

    sim = Simulator('tests/config.yaml')
    print_system_prompt(sim, app_agents=['intro_assessment', 'concept_explanation', 'problem_solving'])
    
    messages = []
    state = 'intro_assessment'
    while (msg_limit := message_limit(messages)).score < 3: # 0: score, 1: result
        if state == 'intro_assessment':
            if (intro_check := check_introduction_assessment(messages)).result:
                state = router(messages)
            else:
                sim.sim_agent.process_turn(messages)
                sim.app_agents['intro_assessment'].process_turn(messages)

        if state == 'concept_explanation':
            if (concept_check := check_concept_explanation(messages)).result:
                state = router(messages)
            else:
                sim.sim_agent.process_turn(messages)
                sim.app_agents['concept_explanation'].process_turn(messages)
        
        if state == 'problem_solving': break

    print_chat(messages)

if __name__ == '__main__':
    main()
