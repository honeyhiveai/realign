
RATING_5_STAR: str = \
'''[Instruction]

You must rate the response on a scale of 1 to 5 and provide a string explanation by strictly following this JSON format: '{"explanation": explanation, "rating": rating, }, for example: '{"explanation": "explanation", "rating": 2}.

Please act as an impartial judge and evaluate the professionalism of the messages provided below. Your evaluation should be based on the mentioned criteria. 

[Criteria]

{{criteria}}

[The Start of Input]

{{messages}}

[The End of Input]
[Evaluation With Rating in JSON format]
'''


SYNTH_USER_PROMPT_GENERATOR_TEMPLATE = \
'''[Instruction]

As an LLM Agent instructor, you must design an instruction for a prudent Synthetic User Agent who is chatting with an AI with this objective to test it out.

[APP OBJECTIVE]
{{app}}

The Synthetic User Agent is someone who wants to {{scenario}}. Your Synthetic User Agent will use a customer-facing application to test some features related to the scenario. Your instruction prompt should be creative and interesting to generate a diverse set of responses from the synthetic user agent for the scenario.

Command your Synthetic User Agent in instruction format, starting with: 'Pretend that you are {{persona}}. You are intelligent, curt, and direct human.'

Respond ONLY with your generated prompt in the following JSON format: {'synth_user_system_prompt': GENERATED_USER_PROMPT}, for example {'synth_user_system_prompt': 'Pretend that you are...'}.
'''