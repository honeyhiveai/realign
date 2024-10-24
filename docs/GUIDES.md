# Guides

## ELO Rating for Content Generation

Code + Config: [tweetbot_elo](../tutorials/tweetbot_elo)

Let's say you want to get the **best** tweet generated by your agent. Given a bunch of generations, you might want to sort the tweets based on quality.

However, LLM judges like the [tweet_judge](TUTORIALS.md#eval-4-tweet-should-score-highly-on-specified-criteria-llm-call) will not perform well on this task since there would be too much data, which would confuse the evaluator.

One solution to this is to have an LLM compare Tweet A vs Tweet B, and select the better one based on some given selection criteria. If we do this for all pairs A, B in the generations, we'll get a list of pairs of tweets, like [(A, B), (A, C), (B, A), (B, C), (C, A), (C, B)]. This will be on the order of N permute 2, or N * (N - 1). 

Next, we make N * (N-1) requests to an LLM judge to tell us the winner of the 2 tweets to get the winner for each pairwise eval.

This is exactly the logic of the out-of-the-box `allm_pairwise_judge` evaluator. This evaluator's first param is `choices`, containing all the generations you want to compare. The response is a list of (winner, loser) pairs.

Now that we have a pairs of (winner, loser), we want to use these to create some kind of ranking. For this, we can use Elo Ratings.


> **Elo Ratings** are a rating system used to calculate the relative skill levels of players in two-player games, most famously in chess.

In this case, we can plug in Realign's out-of-the-box `elo_rating` evaluator to aggregate the values from the `allm_pairwise_judge`. This will return a list of pairs of (tweet, rating) sorted in descending order of rating. 

The highest rated tweet is the overall winner, and the lowest rated tweet the overall loser.


```yaml
evaluators:
  llm_tweet_choice_judge:
    wraps: allm_pairwise_judge
    criteria: |
      - Make sure sentences are concise and don't use flowery language.
      - It should say something interesting to an AI developer.
      - It should make a profound and true claim about AI.
      - It shouldn't sound too salesy or promotional.
      - It should be factually accurate.
      - It should be grammatically correct.
      - Don't use too many adjectives or adverbs.
      - Don't make general claims.
      - Don't start with a question.

    aggregate: elo_ratings(values)
```


## Simulating Conversations with Synthetic Users

Code + Config: [chat_simulation](../tutorials/chat_simulation/)

Let's say you want to test your chatbot by simulating conversations. Since your application is multi-turn, it becomes hard to keep entering user inputs. It might be easier to create synthetic users to test various paths.

In our example, we create a tutor agent which will help a student. The tutor agent has a straitforward setup.

For the synthetic student, we create a `synth_student_agent_generator` which generates system prompts for the synthetic student. Here, we can use the built-in `synthetic_user_prompt_generator` template, which takes the params:
- `app`: the objective of the application, which we reference to the system prompt of the `tutor_agent`
- `persona`: the persona of your app's user
- `scenario`: the particular scenario or app path you want to test

The template will return a JSON containing the prompt for the synthetic user. 

Since we want this prompt to be generated before each run, we can add it to the `before_each` hook and store it in the `run_context`. We can then access this during the simulation `main` to process the app and user turns in the main conversation loop.


```yaml
llm_agents:
  tutor_agent: &tutor_agent
    model: openai/gpt-4o-mini
    system_prompt: &app_objective |
      As an AI tutor, your role is to guide student learning across various subjects through explanations and questions.

      Assess student knowledge and adapt your approach accordingly, providing clear explanations with simple terms and examples.

      Keep responses very concise (maximum 1 sentence) and to the point, and avoid jargon or complex terms.

  synth_student_agent_generator:
    model: openai/gpt-4o-mini
    json_mode: on
    template: synthetic_user_prompt_generator
    template_params:
      app: *app_objective
      persona: math major at Columbia University
      scenario: wants to learn about clustering
```

