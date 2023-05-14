import os
import openai


openai.api_key = ('YOUR_OPENAI_API_KEY_HERE')
# openai.api_key = os.getenv("OPENAI_API_KEY", "")

def gpt_completion(prompt):
	res = openai.Completion.create(model="text-davinci-002", prompt=prompt, max_tokens=500, temperature=1)
	return res["choices"][0]["text"].strip()

