from huggingface_hub.inference_api import InferenceApi

API_TOKEN = "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" #os.environ["API_TOKEN"]
inference = InferenceApi(repo_id="bert-base-uncased", token=API_TOKEN)
#print(inference(inputs="The goal of life is [MASK]."))

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

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query({
    "inputs": "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
})

print(output)