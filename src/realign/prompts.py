def resolve_prompt_template(template_name: str):
    if template_name == 'rating_5_star':
        return RATING_5_STAR
    elif template_name == 'classification':
        return CLASSIFICATION
    elif template_name == 'synthetic_user_prompt_generator':
        return SYNTH_USER_PROMPT_GENERATOR_TEMPLATE
    raise ValueError("Template not found")


RATING_5_STAR: str = \
'''[Instruction]

You must rate the response on a scale of 1 to 5 and provide a string explanation by strictly following this JSON format: '{"explanation": explanation, "rating": rating, }, for example: '{"explanation": "explanation", "rating": 2}.

Please act as an impartial judge and evaluate the messages provided below. Your evaluation should be based on the mentioned criteria. 

[Criteria]

{{criteria}}

[The Start of Input]

{{messages}}

[The End of Input]
[Evaluation With Rating in JSON format]
'''


CLASSIFICATION: str = \
'''[Instruction]

You must classify the following messages into one and exactly one of the classes given, and provide a string explanation for your decision by strictly following this JSON format: '{"explanation": explanation, "class": class, }, for example: '{"explanation": "explanation", "class": CLASS}.

Please act as an impartial judge and classify the messages provided below. Your evaluation should be based on the mentioned criteria. 

[Criteria]

{{criteria}}

[Classes]

{{classes}}

[The Start of Input]

{{messages}}

[The End of Input]
[Evaluation With Class in JSON format]
'''



SYNTH_USER_PROMPT_GENERATOR_TEMPLATE = \
'''[Instruction]

As an LLM Agent instructor, you must design an instruction for a prudent Synthetic User Agent who is chatting with an AI with this objective to test it out.

[APP OBJECTIVE]
{{app}}

The Synthetic User Agent is someone who wants {{scenario}}. Your Synthetic User Agent will use a customer-facing application to test some features related to the scenario. Your instruction prompt should be creative and interesting to generate a diverse set of responses from the synthetic user agent for the scenario.

Command your Synthetic User Agent in instruction format, starting with: 'Pretend that you are {{persona}}. You are intelligent, curt, and direct human. Talk strictly in conversation format. Extremely important: ALL your responses should be ONE sentence only and no more. Start by introducing yourself and stating what you'd like to do.' followed by detailed instructions on how to proceed with the scenario. 

Respond ONLY with your generated prompt in the following JSON format: {'synth_user_system_prompt': GENERATED_USER_PROMPT}, for example {'synth_user_system_prompt': 'Pretend that you are...'}.
'''