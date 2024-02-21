import requests

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query({
    "inputs": "whats capital of france? ",
})

print(output[0]['generated_text'])