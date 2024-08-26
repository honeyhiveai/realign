from realign.simulation import ChatSimulation, Context
from realign.llm_utils import (
    allm_messages_call, 
    str_msgs, 
    print_run_id, 
    print_system_prompt
)

import realign
realign.config.path = 'config.yaml'

class TutorSimulation(ChatSimulation):
    
    async def before_each(self, run_context: Context):
        
        # generate a system prompt for the synthetic user before each run
        new_message = await allm_messages_call(
            agent_name='synth_student_agent_generator',
        )
        run_context.synth_user_system_prompt = new_message.content['synth_user_system_prompt']
        
        # print utilities
        print_run_id(run_context.run_id)
        print_system_prompt(run_context.synth_user_system_prompt, 'user')
        
    async def main(self, run_context: Context):
        
        messages = []
        max_messages = self.max_messages or 3
        
        if self.first_turn_role == 'user' and max_messages > 0:
        
            new_message = await allm_messages_call(
                system_prompt=run_context.synth_user_system_prompt,
                role='user',
                messages=messages
            )
            messages.append(new_message)
            
            # print utilities
            print_run_id(run_context.run_id)
            print(str_msgs(messages[-1]))

        while True:
            # app turn
            if len(messages) > max_messages: break
            new_message = await allm_messages_call(
                agent_name='tutor_agent',
                messages=messages
            )
            messages.append(new_message)
            
            # print utilities
            print_run_id(run_context.run_id)
            print(str_msgs(messages[-1]))

            # synthetic user turn
            if len(messages) > max_messages: break
            new_message = await allm_messages_call(
                system_prompt=run_context.synth_user_system_prompt,
                role='user',
                messages=messages
            )
            messages.append(new_message)
            
            # print utilities
            print_run_id(run_context.run_id)
            print(str_msgs(messages[-1]))
            
        return messages

sim = TutorSimulation()
sim.run(3)