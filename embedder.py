import os
import chromadb
import requests

from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings
)

class CustomEmbedder(EmbeddingFunction):
    def __init__(self) -> None:
        self.API_TOKEN = "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" #"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" #os.environ["API_TOKEN"]
        self.API_URL = "https://c9ejquzh6yum3xqf.us-east-1.aws.endpoints.huggingface.cloud/" #"https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5" #"https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5" #"https://api-inference.huggingface.co/models/BAAI/bge-large-en"
        self.headers = {"Authorization": "Bearer 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"}

    def __call__(self, input: Documents) -> Embeddings:
        rest_client = requests.Session()
        response = rest_client.post(
            #self.API_URL, json={"inputs": input}, headers={"Authorization": f"Bearer {self.API_TOKEN}"}
            self.API_URL, headers=self.headers, json=input
        ).json()
        return response