

from typing import Any, Coroutine
from realign.agents import ChatAgent, SyntheticUserFactory
from realign.types import OpenAIMessage, ModelSettings, RunData
from realign.llm_utils import allm_messages_call, State
from realign.simulation import ChatSimulation, Simulation

class TutorSimulation(ChatSimulation):
    def setup(self):
        self.synthetic_users = SyntheticUserFactory() \
                                .as_a('someone who wants to learn something new') \
                                .they_want_to('learn a new complex subject') \
                                .with_app_objective('learn a new complex subject') \
                                .build_many(self.runs)

        self.app = ChatAgent(system_prompt='Talk to the user seriously.')        
        self.agent = ChatAgent(role='user')

    async def coroutine(self, run_id: int) -> State:
        
        synthetic_user = self.synthetic_users[run_id]
        
        state: State = State()

        state = await self.app.aprocess_turn(state)
        state = await synthetic_user.aprocess_turn(state=state)
        
        print('state', state.messages)

        return state

sim = TutorSimulation().run(3).push_runs_to_dataset('data/run_data.json')







# TODO: printing async chats using 'with'
# instrument agent with Realign
# simulate agent and use evals to improve
# optimize the agent
# publish the results


# see how your chatbot performs on variety of inputs quickly because I cant keep chatting with it
# red teaming / robustness simulations
# query the worst trajectories
