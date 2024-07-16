

rating_5_star = '''
[Instruction]

You must rate the response on a scale of 1 to 5 and provide a string explanation by strictly following this JSON format: '{"explanation": explanation, "rating": rating, }, for example: '{"explanation": "explanation", "rating": 2}.

Please act as an impartial judge and evaluate the professionalism of the messages provided below. Your evaluation should be based on the mentioned criteria. 

[Criteria]

Evaluate whether the messages are professional in nature. Provide a high rating for professional messages and low score for unprofessional messages. Respond with the explanation string and the rating integer in the specified JSON format.

[The Start of Input]
{{messages}}
[The End of Input]
[Evaluation With Rating in JSON format]
'''

synth_user_prompt = ''