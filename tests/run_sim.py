from agentsim import Simulator
from agentsim.evaluators import message_limit, toxicity_score_llm
from agentsim.utils import print_system_prompt, print_chat, evaluations_dataframe

from tests.is_professional_style import is_professional_style

from dotenv import load_dotenv

def main():
    load_dotenv()

    sim = Simulator('tests/config.yaml')
    print_system_prompt(sim)
    
    evaluations = []
    
    messages = []
    while (eval_0 := message_limit(messages)).result: # 0: score, 1: result
        
        # add a new row to the dataframe
        evaluations.append(eval_0)
        
        sim.sim_agent.process_turn(messages)
        sim.app_agent.process_turn(messages)

        eval_1 = is_professional_style(messages)
        evaluations.append(eval_1)

        if eval_1.result == False:
            raise Exception('Professionalism score is below threshold')

        print('toxicity', (eval_2 := toxicity_score_llm(messages[-2:])))
        evaluations.append(eval_2)
    
    print_chat(messages)
    print(evaluations_dataframe(evaluations).head())

if __name__ == '__main__':
    main()
