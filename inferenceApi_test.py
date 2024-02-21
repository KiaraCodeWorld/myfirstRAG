from huggingface_hub.inference_api import InferenceApi

"""
import requests

API_URL = "https://api-inference.huggingface.co/models/valhalla/bart-large-finetuned-squadv1" #"https://api-inference.huggingface.co/models/ESGBERT/EnvironmentalBERT-environmental"
headers = {"Authorization": "Bearer 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"}

question = "What is the population of Jacksonville, Florida?"
context = "As of the most current census, Jacksonville, Florida has a population of 1 million."
#prompt = fUse the following context to answer the question at the end.

{context}

Question: {question}


payload = {
        "inputs": prompt,
        "parameters": { #Try and experiment with the parameters
            "max_new_tokens": 1024,
            "temperature": 0.6,
            "top_p": 0.9,
            "do_sample": False,
            "return_full_text": False
        }
    }

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query({
    "inputs": "whats population of jacksonville?",
})

print(output)
"""

import requests

import requests

import requests

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query({
    "inputs": "Can you please let us know more details about your ",
})

print(output)