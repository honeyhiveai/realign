from realign.types import ModelSettings, OpenAIMessage
from realign.prompts import resolve_prompt_template
from realign.llm_utils import llm_messages_call, allm_messages_call
from typing import Optional, Any, Generator
from abc import abstractmethod
import json
import os
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
    def process_turn(self, state: Any) -> Optional[Any]:
        '''Process a turn and return the updated state'''
        raise NotImplementedError
    
class ChatAgent(AbstractAgent):

    def __init__(self, **model_settings):
        model_settings = model_settings or {'model_settings': ModelSettings(
            model='openai/gpt-4o-mini',
            role='assistant',
        )}

        super().__init__(**model_settings)
        
    async def aprocess_turn(self, messages: list[OpenAIMessage] = []) -> list[OpenAIMessage]:
        '''Process a turn in the conversation'''

        new_message: OpenAIMessage = await allm_messages_call(model_settings=self.model_settings, messages=messages)

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
    
    def __init__(self):
        self.model_settings: ModelSettings = None
        self.system_prompt: str = ""
        self.role: str = ""

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

        return ChatAgent(model_settings=self.model_settings)

class SyntheticUserAgent(ChatAgent):
    
    def __init__(self, **model_settings):
        model_settings = model_settings or {'model_settings': ModelSettings(
            model='openai/gpt-4o-mini',
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
            model='openai/gpt-4o-mini',
            role='user',
            template='synthetic_user_prompt_generator',
            prompt_params={},
            json_mode=True,
            hyperparams={'temperature': 1},
        )
        self.synth_user_system_prompt = None
        self.num_personas = 3
        self.retrieved_personas = None
        
        self.synth_user_model = None
    
    def as_a(self, persona: str) -> 'SyntheticUserBuilder':
        self.persona = persona
        return self
    
    def they_want_to(self, scenario: str) -> 'SyntheticUserBuilder':
        self.scenario = scenario
        return self
    
    def with_app_objective(self, app_objective: str) -> 'SyntheticUserBuilder':
        self.synth_user_builder_model_settings.prompt_params['app'] = app_objective
        return self
    
    def with_system_prompt(self, system_prompt: str) -> 'SyntheticUserBuilder':
        self.synth_user_system_prompt = system_prompt
        return self
    
    def fetch_personas(self) -> 'SyntheticUserBuilder':
        if not self.retrieved_personas or len(self.retrieved_personas) < self.num_personas:
            self.retrieved_personas: list[str] = self.get_personas_from_hub(self.persona)
            self.current_persona_index = 0
            self.persona_generator = self.get_persona_generator()
            print('Retrieved personas:')
            for p in self.retrieved_personas:
                print('-', p)
        return self
    
    def with_synth_user_model(self, model: str) -> 'SyntheticUserBuilder':
        self.synth_user_model = model
        return self
    
    def build(self) -> SyntheticUserAgent:
        
        assert self.retrieved_personas, "Personas must be fetched"
        
        if not self.synth_user_system_prompt:
            if not self.persona:
                raise ValueError("Persona must be set")
            
            # get the next persona
            next_persona = next(self.persona_generator)
            
            # copy the model settings
            self.synth_user_builder_model_settings = self.synth_user_builder_model_settings.copy()
        
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
            self.synth_user_system_prompt = messages[-1].content['synth_user_system_prompt']

        # initialize the synthetic user agent with the generated prompt
        if self.synth_user_model:
            synthetic_user_agent = SyntheticUserAgent(model=self.synth_user_model)
        else:
            synthetic_user_agent = SyntheticUserAgent()
        synthetic_user_agent.model_settings.system_prompt = self.synth_user_system_prompt
        self.synth_user_system_prompt = None

        return synthetic_user_agent

    async def abuild(self, persona_idx: int = 0) -> SyntheticUserAgent:
        assert self.retrieved_personas, "Personas must be fetched"
        assert persona_idx < len(self.retrieved_personas), "Persona index out of range"

        if not self.synth_user_system_prompt:
            if not self.persona:
                raise ValueError("Persona must be set")
            
            # get the next persona
            next_persona = self.retrieved_personas[persona_idx]
            
            # copy the model settings
            settings_copy = self.synth_user_builder_model_settings.copy()
        
            # generate the synthetic user prompt
            settings_copy.prompt_params = {
                **settings_copy.prompt_params,
                'scenario': self.scenario,
                'persona': next_persona,
            }

            prompt_renderer_agent = ChatAgent(model_settings=settings_copy)
            messages: list[OpenAIMessage] = await prompt_renderer_agent.aprocess_turn()
            if len(messages) == 0:
                raise ValueError("No messages generated")
            system_prompt = messages[-1].content['synth_user_system_prompt']
        else:
            system_prompt = self.synth_user_system_prompt
            self.synth_user_system_prompt = None

        # initialize the synthetic user agent with the generated prompt
        if self.synth_user_model:
            synthetic_user_agent = SyntheticUserAgent(model=self.synth_user_model)
        else:
            synthetic_user_agent = SyntheticUserAgent()
        synthetic_user_agent.model_settings.system_prompt = system_prompt
        
        print('Built synthetic user', persona_idx + 1)

        return synthetic_user_agent
    
    async def abuild_many(self, n: int) -> list[SyntheticUserAgent]:
        # Create n agents concurrently
        agents = await asyncio.gather(*[self.abuild(persona_idx) for persona_idx in range(n)])
        return agents

    def get_persona_generator(self) -> Generator[str, None, None]:
        while True:
            yield self.retrieved_personas[self.current_persona_index]
            self.current_persona_index = (self.current_persona_index + 1) % len(self.retrieved_personas)
    
    def with_num_personas(self, num_personas: int) -> 'SyntheticUserBuilder':
        self.num_personas = num_personas
        return self

    def get_personas_from_hub(self, persona: str) -> list[str]:

        try:
            from llama_index.core import VectorStoreIndex, Document, load_index_from_storage
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.core.storage import StorageContext
            from llama_index.core import Settings
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError:
            raise ImportError("llama_index not installed")
        
        persist_dir = os.path.join(os.path.dirname(__file__), "persona-hub/cache")
        personas_path = os.path.join(os.path.dirname(__file__), "persona-hub/persona.jsonl")
        
        # use a smaller model for faster embeddings
        # reduce the dimensions to 512
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small", dimensions=512
        )

        # raise if OPENAI_API_KEY is not set in your environment
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set in environment. It is used to generate personas.")
        
        # check if directory exists
        if not os.path.exists(persist_dir):
            documents = []
            with open(personas_path) as f:
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
            similarity_top_k=self.num_personas,
        )
            
        response = retriever.retrieve(persona)
        personas = []
        for r in response:
            personas.append(r.text)
        return personas
