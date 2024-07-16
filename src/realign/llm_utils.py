from realign.types import ModelSettings, OpenAIMessage, RunData, resolve_prompt_template
from realign.evaluation import evaluator
from jinja2 import Template
from typing import Any, Optional, Self, Generator
from litellm import completion, acompletion
import os
import json
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import asyncio


class AbstractAgent:
    def __init__(self, **model_settings):
        if 'model_settings' not in model_settings:
            model_settings = ModelSettings(**model_settings)
        elif type(model_settings['model_settings']) != ModelSettings:
            raise ValueError("model_settings must be of type ModelSettings")
        else:
            model_settings = model_settings['model_settings']

        self.model_settings: ModelSettings = model_settings

    @abstractmethod
    def process_turn(self, messages: list) -> Optional[Any]:
        raise NotImplementedError
    
class ChatAgent(AbstractAgent):

    def __init__(self, **model_settings):
        model_settings = model_settings or {'model_settings': ModelSettings(
            model='groq/llama3-8b-8192',
            role='assistant',
        )}

        super().__init__(**model_settings)
        
    async def aprocess_turn(self, messages: list[OpenAIMessage] = []) -> list[OpenAIMessage]:
        '''Process a turn in the conversation'''

        new_message: OpenAIMessage = await allm_messages_call(model_settings=self.model_settings, messages=messages)
        
        # wait for 100ms
        # await asyncio.sleep(0.1)

        new_message.role = self.model_settings.role
        
        messages.append(new_message)
        
        # return the updated state
        return messages
    
    def process_turn(self, messages: list[OpenAIMessage] = []) -> list[OpenAIMessage]:
        '''Process a turn in the conversation'''

        new_message: OpenAIMessage = llm_messages_call(model_settings=self.model_settings, messages=messages)

        new_message.role = self.model_settings.role
        
        messages.append(new_message)
        
        # return the updated state
        return messages

class AgentBuilder:
    '''
    @dataclass
    class ModelSettings:
        model: str
        api_key: Optional[str] = None
        hyperparams: Optional[dict[str, Any]] = None
        prompt_params: Optional[dict[str, str]] = None
        template: Optional[str] = None
        system_prompt: Optional[str] = None
        json_mode: Optional[bool] = False
        role: str = 'assistant'
    '''
    
    def __init__(self):
        self.model_settings = None
        self.system_prompt = ""
        self.role = ""

    def with_model(self, model: str) -> 'AgentBuilder':
        self.model_settings.model = model
        return self

    def with_system_prompt(self, prompt: str) -> 'AgentBuilder':
        self.system_prompt = prompt
        return self
    
    def with_template(self, template: str) -> 'AgentBuilder':
        assert resolve_prompt_template(template), "Template not found"
        self.model_settings.template = template
        return self
    
    def with_prompt_params(self, prompt_params: dict[str, str]) -> 'AgentBuilder':
        self.model_settings.prompt_params = prompt_params
        return self

    def with_role(self, role: str) -> 'AgentBuilder':
        self.role = role
        return self

    def with_hyperparameters(self, hyperparams: dict[str, Any]) -> 'AgentBuilder':
        self.model_settings.hyperparams = hyperparams
        return self

    def build(self) -> ChatAgent:
        if not self.model_settings:
            raise ValueError("Model settings must be set")
        if not self.model_settings.model:
            raise ValueError("Model must be set")
        if not self.system_prompt:
            raise ValueError("System prompt must be set")
        if not self.role:
            raise ValueError("Role must be set")

        self.model_settings.system_prompt = self.system_prompt
        self.model_settings.role = self.role

        return ChatAgent(model_settings=self.model_settings, **self.additional_params)

class SyntheticUserAgent(ChatAgent):
    
    def __init__(self, **model_settings):
        model_settings = model_settings or {'model_settings': ModelSettings(
            model='groq/llama3-8b-8192',
            role='user',
        )}
        self.role = 'user'
        super().__init__(**model_settings)

class SyntheticUserBuilder(AgentBuilder):
    
    def __init__(self):
        super().__init__()
        
        # set the role as user
        self.role = 'user'
        self.persona = None
        self.scenario = None
        self.synth_user_builder_model_settings = ModelSettings(
            model='openai/gpt-4o',
            role='user',
            template='synthetic_user_prompt_generator',
            prompt_params={},
            json_mode=True,
            hyperparams={'temperature': 1},
        )
    
    def initialize_persona_generator(self):
        self.retrieved_personas: list[str] = SyntheticUserBuilder.get_personas_from_hub(self.persona, k=10)
        self.current_persona_index = 0
        self.persona_generator = self.get_persona_generator()
        print('Retrieved personas:', self.retrieved_personas)
    
    def as_a(self, persona: str) -> 'SyntheticUserBuilder':
        self.persona = persona

        self.initialize_persona_generator()

        return self
    
    def they_want_to(self, scenario: str) -> 'SyntheticUserBuilder':
        self.scenario = scenario
        return self
    
    def with_app_objective(self, app_objective: str) -> 'SyntheticUserBuilder':
        self.synth_user_builder_model_settings.prompt_params['app'] = app_objective
        return self
    
    def build(self) -> SyntheticUserAgent:
        
        if not self.persona:
            raise ValueError("Persona must be set")
        
        # get the next persona
        next_persona = next(self.persona_generator)
    
        # generate the synthetic user prompt
        self.synth_user_builder_model_settings.prompt_params = {
            **self.synth_user_builder_model_settings.prompt_params,
            'scenario': self.scenario,
            'persona': next_persona,
        }

        prompt_renderer_agent = ChatAgent(model_settings=self.synth_user_builder_model_settings)
        messages: list[OpenAIMessage] = prompt_renderer_agent.process_turn()
        if len(messages) == 0:
            raise ValueError("No messages generated")
        generated_prompt = messages[-1].content['synth_user_system_prompt']

        # initialize the synthetic user agent with the generated prompt
        synthetic_user_agent = SyntheticUserAgent()
        synthetic_user_agent.model_settings.system_prompt = generated_prompt

        return synthetic_user_agent

    def get_persona_generator(self) -> Generator[str, None, None]:
        while True:
            yield self.retrieved_personas[self.current_persona_index]
            self.current_persona_index = (self.current_persona_index + 1) % len(self.retrieved_personas)
    
    @staticmethod
    def get_personas_from_hub(persona: str, k=10) -> list[str]:

        try:
            from llama_index.core import VectorStoreIndex, Document, load_index_from_storage
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.core.storage import StorageContext 
        except ImportError:
            raise ImportError("llama_index not installed")
        
        persist_dir = "src/realign/persona-hub/cache"
        
        # check if directory exists
        if not os.path.exists(persist_dir):

            documents = []
            with open('src/realign/persona-hub/persona.jsonl') as f:
                for line in f:
                    documents.append(Document(text=json.loads(line.strip())['persona']))
                    if len(documents) % 100 == 0:
                        print('Loaded', len(documents), 'documents')

                    if len(documents) == 1000:
                        break

                print('Document loading complete')

                # create and store the index
                index = VectorStoreIndex.from_documents(documents, show_progress=True)
                index.storage_context.persist(persist_dir=persist_dir)
            
        # load the index
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
        except Exception as e:
            raise ValueError(f"Error loading index: {e}")
        
        # configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=k,
        )
            
        response = retriever.retrieve(persona)
        personas = []
        for r in response:
            personas.append(r.text)
        return personas





class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_system_prompt(model_settings: ModelSettings):
    if model_settings.role == 'user':
        print(bcolors.HEADER + '\nUSER SYSTEM PROMPT\n\n', model_settings.system_prompt, bcolors.ENDC)
    elif model_settings.role == 'assistant':
        print(bcolors.HEADER + '\nASSISTANT SYSTEM PROMPT\n\n', model_settings.system_prompt, bcolors.ENDC)

def print_chat(messages):
    for m in messages:
        if m.role == 'user':
            print(bcolors.OKBLUE + '\n', m.role.upper(), '\n\n', m.content, bcolors.ENDC)
        elif m.role == 'assistant':
            print(bcolors.OKGREEN + '\n', m.role.upper(), '\n\n', m.content, bcolors.ENDC)
        elif m.role == 'system':
            pass
        
def print_run_id(run_id):
    print('-' * 100)
    print('RUN ID:',run_id)
    print('-' * 100)
    
def swap_roles(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    for message in messages:
        if message.role == 'user':
            message.role = 'assistant'
        elif message.role == 'assistant':
            message.role = 'user'
    return messages

def llm_call_get_completion_params(model_settings: ModelSettings, messages: list[OpenAIMessage]) -> dict:
        
    # resolve the prompt
    system_prompt = model_settings.resolve_system_prompt()
    
    # insert the system prompt
    if len(messages) == 0:
        messages = [OpenAIMessage(role='system', content=system_prompt)]
    elif messages[0].role != 'system':
        messages.insert(0, OpenAIMessage(role='system', content=system_prompt))
    else:
        messages[0].content = system_prompt
        
    # swap roles for user
    if model_settings.role == 'user':
        messages = swap_roles(messages)
    
    # get the response format
    response_format = model_settings.resolve_response_mode()
    
    # resolve hyperparams
    hyperparams = model_settings.hyperparams or {}
    
    # resolve api_key
    api_key = None
    if model_settings.api_key:
        os.getenv(model_settings.api_key)
        
    # convert messages to dict
    messages_to_llm = [m.__dict__() for m in messages]
    
    return {
        'model': model_settings.model,
        'api_key': api_key,
        'messages': messages_to_llm,
        'response_format': response_format,
        **hyperparams,
    }
    
def llm_call_post_process_response(model_settings: ModelSettings, messages: list[OpenAIMessage], response: Any) -> Any:
    
    # unswap roles for user
    if model_settings.role == 'user':
        messages = swap_roles(messages)

    # process the message
    raw_message = response.choices[0].message
    response_message = OpenAIMessage(role=raw_message['role'], content=raw_message['content'])
    if model_settings.json_mode:
        response_message.content = json.loads(response_message.content)

    return response_message

def llm_messages_call(model_settings: ModelSettings, messages: list[OpenAIMessage] = []) -> OpenAIMessage:
    '''Make an LLM call with the messages provided'''

    # get the params
    params = llm_call_get_completion_params(model_settings, messages)

    # call the LLM
    response = completion(**params)
    
    # post process the response
    message: OpenAIMessage = llm_call_post_process_response(model_settings, messages, response)
    
    return message

async def allm_messages_call(model_settings: ModelSettings, messages: list[OpenAIMessage] = []) -> OpenAIMessage:
    '''Make an LLM call with the messages provided'''

    # get the params
    params = llm_call_get_completion_params(model_settings, messages)

    # call the LLM
    response = await acompletion(**params)
    
    # post process the response
    message: OpenAIMessage = llm_call_post_process_response(model_settings, messages, response)
    
    return message


class Dataset:
    # TODO: async for to validate and load large datasets
    
    @staticmethod
    def validate_data_format(data) -> bool:
        if not data:
            raise ValueError("No data found in the dataset")

        # data must be a dictionary
        if type(data) != dict:
            raise ValueError("Dataset must be a dictionary")

        # data must have inputs, outputs, gtound_truth and metadata keys
        for key in ['inputs', 'outputs', 'ground_truths', 'metadata']:
            if key not in data:
                raise ValueError(f"Dataset must have a '{key}' key")
        
        return True

    def __init__(self, file_path: str):
        self.data = None
        if '.json' not in file_path and '.csv' not in file_path:
            raise ValueError("Dataset file must be a json or csv file")

        if '.json' in file_path:
            with open(file_path) as f:
                data = json.load(f)
                if Dataset.validate_data_format(data):
                    self.data = data

class ChatDataset(Dataset):
 
    def validate_and_load_chat(self) -> list[OpenAIMessage]:

        # load each messages in the ground truth
        for i in range(len(self.data['ground_truths'])):

            # ground_truth must be a dictionary
            if type(self.data['ground_truths'][i]) != dict:
                raise ValueError("Ground truth must be a dictionary")
            
            # ground truth must have messages key
            if 'messages' not in self.data['ground_truths'][i]:
                raise ValueError("Ground truths must have a 'messages' key")

            messages = []
            for message in self.data['ground_truths'][i]['messages']:
                if 'role' not in message or 'content' not in message:
                    raise ValueError("Each message in the ground truth must have a 'role' and 'content' key")
                messages.append(OpenAIMessage(role=message['role'], content=message['content']))
            self.data['ground_truths'][i]['messages'] = messages

        return True
 
    def __init__(self, file_path: str):
        super().__init__(file_path) # sets self.data
        self.validate_and_load_chat() # validates and loads the chat data into self.data

class RunnableProcess(ABC):
    def __init__(self):
        # the subroutine to run
        self.subroutine = None
        
        self.dataset: Dataset = None
        self.app: AbstractAgent = None
        self.simulator: AbstractAgent = None
        self.evaluators: list[callable] = []

    def run(self) -> None:
        load_dotenv()

    async def subroutine(self, run_id: int) -> RunData:
        raise NotImplementedError

    @abstractmethod
    def show_result(self) -> None:
        raise NotImplementedError

class Evaluation(RunnableProcess):
    def run(self) -> None:
        # Implementation for evaluation run
        pass

    def show_result(self) -> None:
        # Implementation for showing evaluation results
        pass

class Simulation(RunnableProcess):

    def __init__(self, subroutine: Any, runs: int = 1):
        super().__init__()
        # simulation params
        self.subroutine = subroutine
        self.runs = runs

        # simulation components
        self.dataset: ChatDataset = None
        self.app: ChatAgent = None
        self.simulator: SyntheticUserBuilder = None
        self.evaluators: list[callable] = []

        # results
        self.run_data: dict[int, RunData] = dict()
        self.eval_results: dict[int, Evaluation] = dict()

    async def subroutine(self, run_id: int) -> RunData:
        raise NotImplementedError("Simulation subroutine must be defined")

    # llms as statisticians berkeley
    async def subroutine_with_evals(self, run_id: int, **subroutine_kwargs) -> Any:
        
        if not self.subroutine:
            raise ValueError("Simulation subroutine must be defined")
        
        # run the simulation subroutine
        final_state = await self.subroutine(run_id, **subroutine_kwargs)

        # wrap the simulation run as an object
        sim_run_data = RunData(final_state, run_id=run_id)
        
        # save the run data
        self.run_data[run_id] = sim_run_data
        
        # run the evaluators
        eval_tasks = []
        for eval_func in self.evaluators:
            # pass object reference to the @evaluator decorator
            eval_tasks.append(asyncio.create_task(eval_func(sim_run_data)))

        # await all the evaluators
        evals: list[Evaluation] = await asyncio.gather(*eval_tasks)
        
        # save the evaluation results
        self.eval_results[run_id] = evals

    # returns a reference to itself to chain more methods
    def run(self, synthetic_user_builder: SyntheticUserBuilder) -> Self:
        # Implementation for simulation run
        super().run()

        # create an asyncio loop
        loop = asyncio.get_event_loop()

        # run the simulation subroutine self.runs times
        app_objective = self.app.model_settings.resolve_system_prompt()

        synth_users = [synthetic_user_builder.with_app_objective(app_objective).build() for _ in range(self.runs)]

        tasks = [self.subroutine_with_evals(run_id, synth_user_agent=synth_users[run_id]) for run_id in range(self.runs)]
        loop.run_until_complete(asyncio.gather(*tasks))
        
        return self
    
    def export_run_data(self) -> dict:
        raise NotImplementedError("Simulation export_run_data must be defined")
    
    def export_eval_results(self) -> dict:
        # {'run_data_hash': [], eval_name': [], 'metadata': [], 'score': [], 'result': []}
        data_obj = {'run_data_hash': [], 'metadata': [], 'evaluations': []}
        for run_id, evals in self.eval_results.items():
            data_obj['run_data_hash'].append(self.run_data[run_id].compute_hash())
            for evaluation_obj in evals:
                data_obj['evaluations'].append({
                    evaluation_obj.eval_name: {
                        'score': evaluation_obj.score,
                        'result': evaluation_obj.result
                    }
                })
        return data_obj
    
    def push_runs_to_dataset(self, dataset_path: str) -> None:
        # adds the results of the run to a new dataset
        with open(dataset_path, 'w') as f:
            json.dump(self.export_run_data(), f)

    def push_evals_dataset(self, evaluations_path: str) -> None:
        
        # adds the evaluations of a run to a new dataset
        with open(evaluations_path, 'w') as f:
            json.dump(self.export_eval_results(), f)

    def show_result(self) -> None:
        # Implementation for showing simulation results
        pass

class ChatSimulation(Simulation):
    '''Responsible for simulating, maintaining, processing states'''
    
    def export_run_data(self) -> dict:
        return_obj = {'inputs': [], 'outputs': [], 'ground_truths': [], 'metadata': []}        
        for run_id, run_data in self.run_data.items():
            return_obj['outputs'].append({'messages': [m.__dict__() for m in run_data.final_state]})
            return_obj['metadata'].append({'run_id': run_id, 'run_data_hash': run_data.compute_hash()})
        return return_obj
    
    async def chat_simulation_subroutine(self, run_id: int, synth_user_agent: SyntheticUserAgent = None) -> list[OpenAIMessage]:
        
        if self.app is None or self.simulator is None:
            raise ValueError("App and simulator agents must be defined")
        elif type(self.app) != ChatAgent or type(synth_user_agent) != SyntheticUserAgent:
            raise ValueError("App and synth_user_agent must be of type ChatAgent")

        print_run_id(run_id)
        print_system_prompt(self.app.model_settings)
        print_system_prompt(synth_user_agent.model_settings)
        
        max_messages = self.max_messages
        
        messages = []
        if self.first_turn_role == 'user' and max_messages > 0:
            messages = await synth_user_agent.aprocess_turn(messages)

        while True:
            if len(messages) > max_messages: break
            messages = await self.app.aprocess_turn(messages)
            print_run_id(run_id)
            print_chat([messages[-1]])

            if len(messages) > max_messages: break
            messages = await synth_user_agent.aprocess_turn(messages)
            print_run_id(run_id)
            print_chat([messages[-1]])
        
        return messages

    def __init__(self,
            subroutine: Any = None, 
            runs: int = 1,
            max_messages: int = 3):

        if not subroutine:
            self.subroutine = subroutine = self.chat_simulation_subroutine

        super().__init__(subroutine, runs)
        
        self.max_messages = max_messages
        self.first_turn_role = 'user'

        if not self.app:
            self.app = ChatAgent()
        
        if not self.simulator:
            self.simulator = SyntheticUserBuilder()

    def run(self) -> Self:
        
        # Implementation for chat simulation run
        return super().run(synthetic_user_builder=self.simulator)
            
    def show_result(self) -> None:
        # Implementation for showing chat simulation results
        pass


# unit test
if __name__ == '__main__':
    
    @evaluator
    def length_evaluator(messages):
        print('Length evaluator', len(messages))
        return len(messages), True
    
    @evaluator(repeat=3)
    async def llm_debate_winner(messages):
        
        model_settings_base = ModelSettings(
            model='openai/gpt-4o',
            role='assistant',
            template='rating_5_star',
            json_mode=True,
        )
        trump_settings = model_settings_base
        model_settings_base.prompt_params = {'criteria': 'Rate Trump\'s performance vs Biden in this debate.'}
        response_trump = await allm_messages_call(model_settings=trump_settings, messages=messages)
        
        biden_settings = model_settings_base
        model_settings_base.prompt_params = {'criteria': 'Rate Biden\'s performance vs Trump in this debate.'}
        
        response_biden = await allm_messages_call(model_settings=biden_settings, messages=messages)
        
        trump_score_vs_biden = response_trump.content['rating'] - response_biden.content['rating']
        print('Trump score vs Biden:', trump_score_vs_biden)
        return trump_score_vs_biden, trump_score_vs_biden >= 0
    
    @evaluator
    def user_role_counter(messages):
        user_messages = [m for m in messages if m.role == 'user']
        print('user role counter', len(user_messages))
        return len(user_messages), True
    
    
    # build a synthetic user
    synth_user_builder = SyntheticUserBuilder().as_a('artistic chef').they_want_to('improve their money management skills')
    
    simulation = ChatSimulation(runs=5, max_messages=5)

    simulation.app = ChatAgent(system_prompt='''
As an AI tutor, your role is to guide student learning across various subjects through explanations and questions. Assess student knowledge and adapt your approach accordingly, providing clear explanations with simple terms and examples. Encourage critical thinking, offer step-by-step problem-solving guidance, and give constructive feedback. Be flexible in addressing different learning styles while maintaining a friendly, encouraging tone. Focus on academic subjects, promote understanding over mere answer-giving, and admit knowledge limitations when necessary. Ensure safe, appropriate interactions and tailor your language to the student's age and level. Your goal is to support learning without replacing human teachers or doing the student's work for them.
        ''',
        model='groq/llama3-8b-8192', role='assistant')
    
    simulation.simulator = synth_user_builder

    # simulation.dataset = ChatDataset('src/realign/data.json')
    # print(simulation.dataset.data)
    
    # add to new dataset
    # continue trajectory
    # continue dataset
    # regenerate a similar dataset
    
    # evaluation.evaluators = [length_evaluator, user_role_counter, llm_debate_winner]
    # simulation.evaluators = [length_evaluator, user_role_counter]

    simulation.run()
    simulation.push_runs_to_dataset('src/realign/run_data.json')
    simulation.push_evals_dataset('src/realign/eval_data.json')
    