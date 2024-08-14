from realign.agents import ChatAgent, SyntheticUserFactory
from realign.llm_utils import State
from realign.simulation import ChatSimulation

class TutorSimulation(ChatSimulation):
    
    async def setup(self, runs):
        self.synthetic_users = await SyntheticUserFactory() \
                                .as_a('someone who wants to learn something new') \
                                .they_want_to('learn a new complex subject') \
                                .with_app_objective('learn a new complex subject') \
                                .abuild_many(runs)

        self.app = ChatAgent(system_prompt='Talk to the user seriously.')
        self.agent = ChatAgent(role='user')

    async def coroutine(self, run_id: int) -> State:
        
        # synthetic_user = self.synthetic_users[run_id]
        
        state: State = State()
        
        state = await self.app.aprocess_turn(state)
        state = await self.agent.aprocess_turn(state)
        
        print('state', state.messages)

        return state

sim = TutorSimulation()
sim.run(3)
sim.push_runs_to_dataset('data/run_data.json')


# TODO
# how to make your sync code async
# make template params easier to use
# YAML configs
# async utilities
# trick with example in comments at the bottom as copilot guide