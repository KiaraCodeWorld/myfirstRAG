import requests

API_URL =  "https://c9ejquzh6yum3xqf.us-east-1.aws.endpoints.huggingface.cloud/" #"https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5" #"https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
headers = {"Authorization": "Bearer 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"} #{"Authorization": "Bearer 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"}

#API_URL =  "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5" #"https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
#headers = {"Authorization": "Bearer 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"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json={"inputs":payload})
    return response.json()

prompt = {
    "inputs": "Today is a sunny day and I will get some ice cream.",
    "member": "234545"}

prompt2 = ["Today is a sunny day and I will get some ice cream."]
prompt3 = "Today is a sunny day and I will get some ice cream."
prompt4 = ["Today ","is cream"]

import json
data = json.dumps(prompt4)

output = query(prompt)
print(output)