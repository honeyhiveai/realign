# agentsim

`agentsim` is a simulation based evaluation framework for multi-step agents.

## what is `agentsim`?
`agentsim` helps you iterate by:
1. Generating insightful synthetic trajectories for testing
2. Evaluating the trajectories to select the one with most interesting results
3. Boostrapping your Agent architecture for rapid development

## why build `agentsim`?
there are many LLM eval frameworks that are already out there.
so why are we building another one?

1. **Synthetic Dataset Generation Toolkit**

AI Applications are too open ended. Customers don't know what they want. So we need to simulate and understand where our application breaks.

No framework exists for generating synthetic trajectories. Since agents trajectories are variable, we help you increase your test surface area using synthetic trajectories that you can customize with your ideal user persona.

2. **Maximizing Developer Attention**

AI Applications can only be aligned by observing the most anomalous trajectories.

We use statistics to analyze multiple trajectories per run and show you the most interesting one first. Test cases that deviate most from the others or trigger/break evaluations are more interesting for development.

3. **Composable Abstractions**

Current frameworks lack the base abstractions to evaluate trajectories and also make it hard to customize the evaluators.

For the beginner, we offer several boostrapped stacks for chatbots and agents.
For the novice, we offer customization in Evaluators, App Settings, and Trajectory generators.
For the expert, we allow developers to build their own console and simulation logic.


## Quickstart




## how to use agentsim?

to setup a test, we need to define 3 things:
- **generator**: a function that generates the input data
- **scorer**: a function that scores the output data
- **evaluator**: a function that checks if the scores are in your target range


Tests: Alignment ! 😡
App: Alignment ? 🧐
Evals: Alignment = 😇