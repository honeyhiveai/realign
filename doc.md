# Documentation

# Core Concepts

## Evaluators

### **What is an `evaluator`?**

An evaluator is a function which measures the quality of an LLM application output by giving it a `score` and evaluating a bool `result` which indicates whether if the score is in the target range.

It follows the signature:

```python
from realign.evaluation import evaluator

@evaluator
def customer_sentiment_evaluator(output):
	
	# calculate score of the output
	score = ...
	
	# evaluate whether the score is in the target range
	result = score > target
	
	return score, result
```

The `output` is a key-value pair of various output params. For example, a single step application could have

```python
output = { 'response': 'The capital of France is Paris' } 
```

and a multi step application could have the output:

```python
output = { 'messages': [ 
							{'role': 'user', 'content': 'What is the capital of France?'}, 
							{'role': 'assistant', 'content': 'The capital of France is Paris'}
						]
					}
```

### What is the @evaluator decorator?

The `@evaluator` decorator simply helps you wrap your evaluator output as an object which has some in built helpers. Importantly, it allows you do use your evaluation as a guardrail in your code as follows:

```python
from realign.evaluation import toxicity_llm_eval, topic_classification_eval

messages = []

# ... run your application

evaluation = toxicity_llm_evaluator({'messages': messages})

if evaluation.result == False:
	raise Exception('Output failed quality guardrail.')

print('Score:', evaluation.score)

categories = ['solve a problem', 'explain a concept']

# the unpack() function will return the score and result
score, result = topic_classification_eval({'messages': messages, 
																					'categories': categories}).unpack()

assert result
print('Category:', score) # will print the category
```

### Which evaluators should I use?

Realign offers a library of well-tuned evaluators which you can directly import into your application or test suite. You can also create a custom evaluator by implementing a function with the same format. Evaluators generally come in 3 different flavors:

- Python evaluators (code, API calls, 3rd party libraries, etc)
- LLM as a judge evaluators (using an intelligent model to evaluate)
- NLP evaluators (such as BERT-based evaluators)

## Datasets

You likely fall into 3 scenarios:

1. You have a large dataset covering a variety of test cases
2. You have a small dataset and want to synthetically augment it
3. You don’t have a dataset and want to generate one (simulation)

In Realign, your dataset for single turn agents must follow this schema. Note that `metadata` is optional. The dataset must be a JSON file or CSV file. 

| input | output | ground_truth | metadata |
| --- | --- | --- | --- |
| { “question”: “What is the capital of France?” } |  | {”answer”: “Paris”} | capitals |
|  |  |  |  |

For multi turn agents, you must follow the schema

| input | output | ground_truth | metadata |
| --- | --- | --- | --- |
| {’user_question’: 'What is the capital of France?'} |  | {'messages': [ 
{'role': 'user', 'content': 'What is the capital of France?'}, 
{'role': 'assistant', 'content': 'The capital of France is Paris'}
]} | capitals |

## Evaluations

In Realign, you can easily evaluate your application against dataset and see the scores and results to understand how your application performs. This can help you develop and improve the quality of your application or prevent regression. 

For each of these scenarios, Realign offers simple tools to run your evaluations and get a quality report. 

### Synthetic Data Generation

- params to generate the inputs

```python
dataset = Dataset('file.json')

app = App(prompt_template='hello {{var}}', model_settings=ModelSettings())
app.process_turn(inputs)

# prompt inputs
# rag inputs 
# agent inputs

def prompt_input_generator(app_description, template_params):
	# get context chunks
	# generate questions
	return input, ground_truth: Optional
	
def messages_simulator(personas, scenarios, app, chat_history=None):
	# generate trajectory
	
	synth_user = ...
	
	return input, ground_truth

dataset = Dataset('file.json')

evaluation = Evaluation()
evaluation.dataset = dataset
evaluation.app = app
evaluation.evaluators = []
evaluation.run()
evaluation.show_result()

dataset = Dataset('file.json').synthetic_input_generator(app_description, input_params)
personas, scenarios = Dataset(dataset).yield_personas_scenarios()

simulation = Simulation(subroutine, n=50, turns=20)

simulation.dataset = Dataset(dataset).first_message()
simulation.simulator = SyntheticUser(personas, scenarios, process_turn)
evaluation.app = OnlineAgent(user_endpint='multion/user:5000')
simulation.evaluators = []

simulation.run()
simulation.show_result()

```

### Evaluation DX

```python
evaluation = Evaluation()

evaluation.dataset = dataset
evaluation.app = SingleTurnApp(system_prompt='{{user_question}}', 
															prompt_params=dataset.input)
evaluation.evaluators = [ toxicity_llm_eval, topic_classification_eval ]
evaluation.run()
evaluation.show_results()
```

### Large Dataset Evaluation

```python
from realign.evaluators import toxicity_llm_eval, topic_classification_eval
from realign.datasets import Dataset

dataset = Dataset('data.json')

evaluation = Evaluation()

evaluation.dataset = dataset
evaluation.app = SingleTurnApp(system_prompt='{{user_question}}', 
															prompt_params=dataset.input)

evaluation.evaluators = [ toxicity_llm_eval, topic_classification_eval ]

evaluation.run()

evaluation.show_results()
```

### Augmented Dataset Evaluation

TODO: provide explanation on Augmented Dataset

TODO: diagrams

```python
from realign.evaluators import toxicity_llm_eval, topic_classification_eval
from realign.datasets import Dataset

dataset = Dataset('chats.json')

# continue conversations for 5 more turns by simulating user turns
dataset.continue_trajectory(turns=5, generator=app)

# regenerate conversations of similar lengths and trajectories 
# by simulating user turns
dataset.regenerate_chat(generator=app)

# evaluation
evaluation = Evaluation()
evaluation.dataset = dataset
evaluation.evaluators = [ toxicity_llm_eval, topic_classification_eval ]

evaluation.run()

evaluation.show_results()
```

### Synthetic Dataset

```python
# simulate entire conversations based on personas and scenarios
personas = [
	'''An urban planner looking to understand the distribution and organization 
	of public services in the Halifax Regional Municipality''',
	
	'''A high school literature teacher looking for supplementary materials
	 to enrich their curriculum and provide students with a deeper understanding
	  of their state's cultural heritage.'''
]

scenarios = [
	'''Can you provide me with a list of 5-7 educational resources, including videos, articles, and interactive websites, 
	that explore the literary works and cultural significance of Native American authors from my state, such as Louise Erdrich or 
	Sherman Alexie, to incorporate into my 11th-grade American Literature curriculum and help my students better understand the 
	cultural heritage of our region?''',
	
	'''Compare and contrast the distribution of public services such as libraries, community centers, and public transportation 
	in different neighborhoods of Halifax, and discuss how the municipal government's urban planning strategies impact access to 
	these services for residents of varying socioeconomic backgrounds.'''
]

dataset.simulate(n=50, personas=personas, scenarios=scenarios, turns=20)
```

## Config YAMLs

In your LLM application, you might have settings which are stateless (settings that don’t depend on the runtime state of your application). These are usually things like:

- Application Config
    - Model Config
        - Model name
        - Model provider
        - Model hyperparameters
        - Prompt template to use
        - System prompt
    - Tool Config (Vector Databases, External APIs, etc.)
        - Embedding model to use
        - Vector database provider (Chroma, Pinecone, etc.)
        - Chunk size/overlap
        - Index ID/Name
- Evaluator Config
    - Target passing range of the evaluator
    - Code evaluator settings
        - Required input parameters
    - LLM evaluator settings (model, prompt etc)
        - Required input parameters
        - Model name
        - Model provider
        - Model hyperparameters
        - Prompt template to use
        - System prompt
- Dataset Config
    - Dataset name
    - Number of test cases
    - Dataset source
    - Synthetic generator config
        - Model name
        - Model provider
        - Model hyperparameters
        - Prompt template to use
        - System prompt
- Simulator Config
    - User Simulator Config
        - Model name
        - Model provider
        - Model hyperparameters
        - Prompt template to use
        - System prompt
        - etc. (add more)
    - Environment Simulator Config
- Evaluation Harness Config
    - Dataset Config
    - Evaluator Config
    - Evaluating the composite scores of various evaluators
- Simulation Harness Config
    - Simulator Config
    - Evaluator Config
    - Number of turns to simulate

With increasing complexity in your applications, it might become challenging to keep track of various prompts, settings, and configurations. Since these don’t depend on the runtime state of the applications, we can use Realign configs to specify them as a YAML file. 

clustering by topic in a given state

system prompt builder

khan academy

We are simulating / evaluating. Evaluators are evaluting various states in the system. Datasets are generating various states in the system.

ChatSimulator

- load your dataset
- run evals
- generate inputs to feed in

AppSimulator

- load your dataset
- run evals
- generate inputs to feed in
- use dataset as a seed

RAGSimulator

RAGChatAppSimulator

- define your subroutine which is the full flow
- call arbitrary evals on any part of the state

Generate inputs given a seed and run your pipeline

Agents are doing something in a simulated environment, and we can interpret their activity using evaluators. What sensors do you 

Tuning evaluators:

- get datasets
- run evaluators
- tiny-judge

External environment simulator (later)

- server

Use cases

- integrating with distributed system

Rate limiting

cost and time metrics

Amazing documentation

visualization

Synthetic datasets

- single turn dataset generation
- few shot
- zero shot
- trajectory continuation

Base Simulation harness

- chat

Optimize

- get more feedback
- better docs
- work w Mohak to create a list to send to

Is simulation a good enough sell? vs neater evaluation

release docs + base functionality

Core Concepts

- Evaluations = app + dataset + evaluators
- Simulations = app + dataset + simulator + evaluators
- @evaluations (repeats, metrics etc) + templated evaluators
- Results (eval_data.json, run_data.json)
- Dataset (schema + seed to generate simulator)

Risky Hypotheses

- *viability of few shot and zero shot datasets (reverse engineering synth users, personas + scenarios)*
- trajectory evaluation and visualization tools (state changes across runs)
- distributed systems vs. sync model of a distributed system

Ideas

- leverage persona hub dataset
- simulation composition?
- config
- logit based evaluators