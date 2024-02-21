import os
import requests

# #Set a API_TOKEN environment variable before running
API_TOKEN = "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" #os.environ["API_TOKEN"]
API_URL =  "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2" #"https://api-inference.huggingface.co/models/ESGBERT/EnvironmentalBERT-environmental" #Add a URL for a model of your choosing
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('***** LLM Demo - Inference API ***')







def query2(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def query(prompt):
    payload = {
        "inputs": {
            "question": "What's my name?",
            "context": "My name is Clara and I live in Berkeley.",
        },
        "parameters": { #Try and experiment with the parameters
            "max_new_tokens": 1024,
            "temperature": 0.6,
            "top_p": 0.9,
            "do_sample": False,
            "return_full_text": False
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json() #.json()[0]['generated_text']



question = "What is the population of Jacksonville, Florida?"
context = "As of the most current census, Jacksonville, Florida has a population of 1 million."
prompt = f"""Use the following context to answer the question at the end.
Question: {question}
"""
prompt2 = f"""{{     
            "question": "{question}",
            "context": "{context}",
       }}"""

prompt3 = f"""{{       
            "question": {question},
            "context": {context},
        }}"""

print(prompt2)

print(query(prompt2))