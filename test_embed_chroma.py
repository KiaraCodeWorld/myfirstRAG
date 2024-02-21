import os
import chromadb
import requests

from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings
)

API_TOKEN = "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"  # os.environ["API_TOKEN"]
API_URL = "https://c9ejquzh6yum3xqf.us-east-1.aws.endpoints.huggingface.cloud/"

class CustomEmbedder(EmbeddingFunction):
    def __init__(self) -> None:
        self.API_TOKEN = "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" #os.environ["API_TOKEN"]
        self.API_URL = "https://c9ejquzh6yum3xqf.us-east-1.aws.endpoints.huggingface.cloud/" #"https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5" #"https://api-inference.huggingface.co/models/BAAI/bge-large-en"
        self.headers = {"Authorization": "Bearer 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"}

    def __call__(self, input: Documents) -> Embeddings:
        rest_client = requests.Session()
        response = rest_client.post(
            #self.API_URL, json={"inputs": input}, headers={"Authorization": f"Bearer {self.API_TOKEN}"}
            self.API_URL, headers=self.headers, json=input
        ).json()
        return response


docs1 = {
	"inputs": "Today is a sunny day and I will get some ice cream.",
}
ids = [  "1" ]

dataset = [
    "My cat is named Francis.",
    "I want to visit Italy.",
    "I need to download more RAM."
]
metadatas = [
    {"doc_name": "testdoc"},
    {"doc_name": "testdoc"},
    {"doc_name": "testdoc"}
]
ids = [
    "1",
    "2",
    "3"
]
custom_embedder = CustomEmbedder()
chroma_client = chromadb.PersistentClient(path="../testdb")
#chroma_client.reset()
chroma_client.heartbeat()
chroma_client.delete_collection("test_collection")
collection = chroma_client.create_collection(name="test_collection", embedding_function=custom_embedder)

print(custom_embedder(docs1))

#collection.add(documents=dataset, metadatas=metadatas, ids=ids)

collection.add(
        documents=dataset,
        metadatas=metadatas,
        ids=ids
    )

results = collection.query(
        query_texts=["travel"],
        n_results=2
    )

print(results)