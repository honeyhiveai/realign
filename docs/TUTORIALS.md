# Tutorials

Realign is a tool to test and evaluate your LLM agents blazing fast. We'll provide you with the rools to make 100s of LLM calls in parallel, with ANY model.

[Introduction to LLM Agents](#introduction-to-llm-agents)

1. [Simple Tweet Bot](#simple-tweet-bot)
2. [Generate 10 Tweets in Parallel (Async)](#generate-10-tweets-in-parallel-async)
3. [Using Config Files](#using-config-files)
4. [Set up Evaluators](#set-up-evaluators)
5. [Using Realign Evaluators](#using-realign-evaluators)

### Introduction to LLM Agents

#### Simple Tweet Bot

Let's start by creating a simple Twitter (X?) bot:

```python
from realign import llm_messages_call

def tweetbot(prompt: str) -> str:

    new_message = llm_messages_call(
        messages=[{
            "role": "user", 
            "content": prompt
        }],
    )

    print('\nTweet:\n\n', new_message.content, '\n\n')

tweetbot("Write a tweet about best practices for prompt engineering")
```

You should get something that looks like this:

```
Created model router for openai/gpt-4o-mini
Successfully called openai/gpt-4o-mini in 1.151 seconds.

Tweet:

 🔍✨ Best practices for prompt engineering:

1️⃣ Be specific: Clearly define your objectives.
2️⃣ Iterative testing: Refine prompts based on outputs.
3️⃣ Context is key: Provide relevant background info.
4️⃣ Experiment with formats: Questions, completion, or examples.
5️⃣ Use constraints: Limit scope for focused responses.

#PromptEngineering #AI #MachineLearning 
```

Great! The logs start by telling you which model you are using. The `model router` protects you from rate limits and handles retries.

You must have noticed that the default model is `openai/gpt-4o-mini`. To change your model, simply enter a new model using `model=`.

```python
    new_message = llm_messages_call(
        model='anthropic/claude-3-5-sonnet-20240620',
        messages=[{
            "role": "user", 
            "content": prompt
        }],
    )
```

Realign leverages LiteLLM to help you plug in any model in a single line of code using the format `<provider>/<model>`. You can find their 100+ supported models [here](https://docs.litellm.ai/docs/providers).

#### Generate 10 Tweets in Parallel (Async)

Now let's assume you want to create 10 tweets and see which one looks best.

You could do this by calling `llm_messages_call` 10 times. However, this would be slow. Let's try it:

```python
from realign import llm_messages_call
import time


def tweet(prompt, i):

    new_message = llm_messages_call(
        messages=[{
            "role": "user", 
            "content": prompt
        }],
    )

    print(f'\nTweet {i+1}:\n\n', new_message.content, '\n\n')


def main(prompt):
    for i in range(10):
        tweet(prompt, i)

start_time = time.time()

main("Write a tweet about best practices for prompt engineering")

print(f'Total time: {time.time() - start_time:.2f} seconds')
```

You should see something like:

```
Created model router for openai/gpt-4o-mini
Successfully called openai/gpt-4o-mini in 1.256 seconds.
Successfully called openai/gpt-4o-mini in 1.445 seconds.
Successfully called openai/gpt-4o-mini in 1.449 seconds.
Successfully called openai/gpt-4o-mini in 1.448 seconds.
Successfully called openai/gpt-4o-mini in 1.552 seconds.
Successfully called openai/gpt-4o-mini in 1.554 seconds.
Successfully called openai/gpt-4o-mini in 1.554 seconds.
Successfully called openai/gpt-4o-mini in 1.600 seconds.
Successfully called openai/gpt-4o-mini in 1.633 seconds.
Successfully called openai/gpt-4o-mini in 1.906 seconds.

Tweet 1:

 🚀✨ Best practices for prompt engineering: 

1. **Be Clear

... 10 tweets


Total time: 15.58 seconds
```

It takes about 15 seconds because the API calls happen sequentially. To run these API calls in parallel, you can use the `allm_messages_call` method and `run_async` utility.

```python
from realign import allm_messages_call, run_async
import time


async def tweet(prompt, i):

    new_message = await allm_messages_call(
        messages=[{
            "role": "user", 
            "content": prompt
        }],
    )

    print(f'\nTweet {i+1}:\n\n', new_message.content, '\n\n')


def main(prompt):

    tasks = []
    for i in range(10):
        tasks.append(tweet(prompt, i))

    # run the tasks in parallel
    run_async(tasks)

start_time = time.time()

main("Write a tweet about best practices for prompt engineering")

print(f'Total time: {time.time() - start_time:.2f} seconds')
```

You should see something like:

```
Created model router for openai/gpt-4o-mini
openai/gpt-4o-mini Processing requests...
Successfully called openai/gpt-4o-mini in 1.191 seconds.
Successfully called openai/gpt-4o-mini in 1.374 seconds.
Successfully called openai/gpt-4o-mini in 1.446 seconds.
Successfully called openai/gpt-4o-mini in 1.476 seconds.
Successfully called openai/gpt-4o-mini in 1.478 seconds.
Successfully called openai/gpt-4o-mini in 1.575 seconds.
Successfully called openai/gpt-4o-mini in 1.662 seconds.
Successfully called openai/gpt-4o-mini in 1.944 seconds.
Successfully called openai/gpt-4o-mini in 2.383 seconds.
Successfully called openai/gpt-4o-mini in 2.467 seconds.

Tweet 1:

✨ Mastering prompt engineering? Here are some best practices! ✨

... 10 tweets

Total time: 2.47 seconds
```

The total time is now the time taken by the **longest API call** instead of the total time taken by all calls. This allows you to iterate much quicker!

Here's a summary of the main changes compared to the synchronous run:

- Change the method of `tweet` from `def` to `async def`. 

- Change `llm_messages_call` to `allm_messages_call` which is the async version, and add the `await` keyword before it.

- In the for loop, we add the outputs of the `tweet` calls to a `tasks` list. This contains our 10 scheduled tasks. These tasks are called **coroutines** in Python and are similar to promises in Javascript. More info on how coroutines work [here](#TODO).

- Finally, we use the `run_async` utility which takes a single list of coroutines, and waits until they are all complete. If `tweet` returned something, the `run_async` function would return a list of returns.

### Using Config Files

#### Tweet Bot with Template

You might want the Tweet to follow certain rules and patterns. To do this, we can use a prompt template:

```python
from realign import allm_messages_call, run_async

async def tweet(prompt, i):

    new_message = await allm_messages_call(
        messages=[{
            "role": "user", 
            "content": prompt
        }],
    )

    print(f'\nTweet {i+1}:\n\n', new_message.content, '\n\n')


def main(prompt):

    tasks = []
    for i in range(10):
        tasks.append(tweet(prompt, i))

    # run the tasks in parallel
    run_async(tasks)

prompt = "Write a tweet about best practices for prompt engineering"

template = f'''
As an AI language model, your task is to create a high-quality tweet. Follow these guidelines:

1. Length: Keep the tweet within 280 characters, including spaces and punctuation.
2. Purpose: Clearly define the tweet's purpose (e.g., informing, entertaining, inspiring, or promoting).
3. Audience: Consider the target audience and tailor the language and content accordingly.
4. Tone: Maintain a consistent tone and don't sound salesy or disingenuous. Be based and direct.
5. Content: Ensure the tweet is:
    - Relevant and timely
    - Accurate and factual (if sharing information)
    - Original and engaging
    - Clear and concise

6. Structure:
    - Start with a hook or attention-grabbing element
    - Present the main idea or message clearly
    - End with a thought-provoking insight (if appropriate)

7. Hashtags: Include 1-2 relevant hashtags, if applicable, to increase discoverability.
8. Proofread: Ensure there are no spelling or grammatical errors.

After generating the tweet, review it to ensure it meets these criteria and make any necessary adjustments to optimize its quality and potential engagement.

Here is your instruction:

{prompt}
'''

main(template)
```

However, this is somewhat difficult to maintain in a file which should ideally only have your code! Realign uses YAML configuration files to keep track of all settings that are stateless (things that don't depend on the application runtime state).

This can include things like model, system_prompt, template, temperature, etc. 

Let's use a config file for this example. Create a `config.yaml` file in your directory, and paste the following:

```yaml
llm_agents:
  tweetbot:
    model: openai/gpt-4o-mini
    template: |
      As an AI language model, your task is to create a high-quality tweet. Follow these guidelines:

      1. Length: Keep the tweet within 280 characters, including spaces and punctuation.
      2. Purpose: Clearly define the tweet's purpose (e.g., informing, entertaining, inspiring, or promoting).
      3. Audience: Consider the target audience and tailor the language and content accordingly.
      4. Tone: Maintain a consistent tone and don't sound salesy or disingenuous. Be based and direct.
      5. Content: Ensure the tweet is:
          - Relevant and timely
          - Accurate and factual (if sharing information)
          - Original and engaging
          - Clear and concise

      6. Structure:
          - Start with a hook or attention-grabbing element
          - Present the main idea or message clearly
          - End with a thought-provoking insight (if appropriate)

      7. Hashtags: Include 1-2 relevant hashtags, if applicable, to increase discoverability.
      8. Proofread: Ensure there are no spelling or grammatical errors.

      After generating the tweet, review it to ensure it meets these criteria and make any necessary adjustments to optimize its quality and potential engagement.

      Here is your instruction:

      {{prompt}}
```

Some notes on the YAML file:

- The key 'tweetbot' is the agent name. This is referenced in the code using the `agent_name` param.

- The `model` field specifies the model in `<provider>/<model>` format.

- The `template` field is a Jinja template (string with double curly braces for variables, like `{{var}}`). The template is rendered using a Python dictionary which maps the variable key string with the rendered string for that variable name. This is passed into `allm_messages_call` using the `template_param` field.

You can find the full specification for the YAML config [here](./CONCEPTS.md#agents).


We can use this agent in our code by modifying the `allm_messages_call` params:

```python
async def tweet(prompt, i):

    new_message = await allm_messages_call(
        agent_name='tweetbot',
        template_params={'prompt': prompt}
    )

    print(f'\nTweet {i+1}:\n\n', new_message.content, '\n\n')
```

Finally, we just call `main` with our prompt:

```python
main("Write a tweet about best practices for prompt engineering")
```

Running this will use the config file settings to run your agent. To make changes to the agent's config, you can make changes to the agent config directly. This allows you to quickly play with various settings including:

- model (string)

- hyperparams (model-specific dict)

- system_prompt (string)

- template (string)

- template_params (key-value map)

- json_mode (bool)

- role ('assistant' / 'user')

For example, we can set the temperature to 1 by adding the following line:

```yaml
llm_agents:
  tweetbot:
    model: openai/gpt-4o-mini
    hyperparams:
      temperature: 1.0
    template: |
      As an AI language model, ...
```

### Set up Evaluators with Config Files

Now that you have a Tweet agent that can generate tweets quickly, you might want to evaluate the responses to get make sure the quality is high. To do this, we can set up evaluators.

But what is an evaluator?

> An **Evaluator** is a function which *scores* your app's output and *checks* if the score is within a *target* range.

There are a few ways to implement Evaluators:

- Python Evaluator: simple (or complex) python functions.

- Local Model Evaluator: these use a model on your local device

- LLM Evaluator: these use an LLM to generate a score for your application


> In Realign, *ANY* Python function can become an evaluator if it is decorated with **@evaluator** (or **@aevaluator** if function is async). A Realign evaluator's kwargs, repeat, targets and other stateless settings can be configured in the YAML file using the funciton name as the key. 


Let's build 4 evaluators:

1. Tweet should be under 280 characters (simple Python function)

2. Tweet should not be hateful (Hugginface model)

3. Tweet should be of positive sentiment (Hugginface model)

4. Tweet should score highly on specified criteria (LLM call)

#### Eval 1: Tweet should be under 280 characters

To evaluate this metric, we simply measure the Python len() and assert that we are within the character limit. Pretty simple.

```python
def tweet_length_checker(tweet_text):
    # Evaluation = assert that score(output) within target range

    num_chars = len(tweet_text) # score(output)

    # assert that score is within target range
    assert 0 < num_chars <= 280

    # Evaluation Result
    return eval_result
```

#### Eval 2: Tweet should not be hateful

For this metric, we can use one of Huggingface's Text Classification models. Here's a function you can use: 

First install

```bash
pip install transformers datasets evaluate accelerate torch
```

Then paste

```python
from transformers import pipeline
import torch

def hate_speech_detection(tweet_text):
    # Evaluation = assert that score(output) within target range
    # score = hate / nothate
    # target = nothate

    # Check if MPS is available (Mac Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) device for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")

    # Create the pipeline with the specified device
    pipe = pipeline(task='text-classification', 
                    model='facebook/roberta-hate-speech-dynabench-r4-target',
                    device=device)

    # get the response
    response = pipe(tweet_text)

    assert response[0]['label'] == 'nothate'

    return response[0]['label']
```

This will return something like

```python
[{'label': 'nothate', 'score': 0.9998}]
```

#### Eval 3: Tweet should have positive sentiment

Similar to the last example, we can use Huggingface pipelines:

```python
from transformers import pipeline
import torch

def sentiment_detection(tweet_text):
    # Evaluation = assert that score(output) within target range
    # score = positive / neutral / negative
    # target = [positive, neutral]

    # Check if MPS is available (Mac Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) device for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")

    # Create the pipeline with the specified device
    pipe = pipeline(task='text-classification', 
                    model='cardiffnlp/twitter-roberta-base-sentiment-latest',
                    device=device)

    # get the response
    response = pipe(tweet_text)


    assert response[0]['label'] in ['positive', 'neutral']

    return response[0]
```

Feeling icky with all the code duplication? In Realign, we can pull out all the static / stateless settings into the config file. This helps you cleanly manage your evaluators. Moreover, Realign provides in-built configurations and evaluators.

&nbsp;
&nbsp;
&nbsp;


### Using Realign Evaluators

In Realign, any Python function can be converted to an Evaluator using the `@evaluator` decorator. If your function is async, you can use the `@aevaluator` decorator. 

In our case, Realign @evaluator can help you create a wrapper around a base implementation of Huggingface pipeline, abstracting out any hyperparams you like. 

We'll do this in 3 steps:

1. Create the base evaluator with the right keyword args (in this case, task and model)

2. Create a config file and add the Huggingface evaluator settings

3. Use these evaluators in your code

Let's begin:

1. **Create the base evaluator with the right keyword args (in this case, task and model)**

```python
from transformers import pipeline
import torch

from realign.evaluators import evaluator

@evaluator
def hf_pipeline(tweet_text, task=None, model=None):
    assert task and model # task and model should be specified

    # Check if MPS is available (Mac Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) device for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")

    # Create the pipeline with the specified device
    pipe = pipeline(task=task, 
                    model=model,
                    device=device)

    # get the response
    print(f'\n\nRunning hf_pipeline with task {task} and model {model}\n\n')
    response = pipe(tweet_text)

    return response[0]
```

2. **Create a config file and add the Huggingface evaluator settings**

For the config, let's create a file in the same directory called `config.yaml`, and paste the following:

```yaml
evaluators:
    hf_hate_speech:
        wraps: hf_pipeline
        model: facebook/roberta-hate-speech-dynabench-r4-target
        task: text-classification

        checker: value['label'] in target
        target: [nothate]
        asserts: on

    hf_sentiment_classifier:
        wraps: hf_pipeline
        task: text-classification
        model: cardiffnlp/twitter-roberta-base-sentiment-latest

        checker: value['label'] in target
        target: [positive, neutral]
        asserts: on
```

3. **Use these evaluators in your code**

We can update our tweet function as follows:

```python
async def tweet(prompt, i):

    new_message = await allm_messages_call(
        agent_name='tweetbot',
        template_params={'prompt': prompt}
    )

    # evaluate for hate speech
    evaluators['hf_hate_speech'](new_message.content)

    # evaluate for positive sentiment 
    evaluators['hf_sentiment_classifier'](new_message.content)
```

Using the YAML file helps you tweak different hyperparameters and settings easily wihthout changing your code. For example, you can change the model or task, or adjust the target classes. This is a convenient way to adjust hyperparams for various wrappers. Of course, you can manually create your own wrappers as well.

Let's quickly switch to configs for our `tweet_length_checker` evaluator as well:

```python
@evaluator
def tweet_char_count(text: str) -> int:
    return len(text)
```

```yaml
evaluators:
    # ... other evaluators

    tweet_length_checker:
        checker: numrange(value, target)
        target: (0, 280]
        asserts: on
```

#### Eval 4: Tweet should score highly on specified criteria (LLM call)

LLMs have great language and reasoning skills, and so we can use them to evaluate the responses of another LLM.

We can use Realign's `allm_rating_json` evaluator, which is already implemented for you. You can check out its configuration [here](../src/realign/defaults.yaml) and implementation [here](../src/realign/evallib/llm.py). TL;DR: it uses the criteria to evaluate the messages, repeats it 3 times and aggregates the scores and explanations. 

```yaml
tweet_judge:
    wraps: allm_rating_json
    criteria: |
      1. Make sure sentences are concise and don't use flowery language.
      2. It shouldn't sound too salesy or promotional.
      3. Don't use too many adjectives or adverbs.
      4. Don't make general claims.
      5. Don't start with a question.
```
