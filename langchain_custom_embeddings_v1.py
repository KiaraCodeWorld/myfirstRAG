import json
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import requests
from dotenv import load_dotenv
import os
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from dotenv import load_dotenv
from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings
)


class CustomEmbeddings():
    load_dotenv()
    API_KEY = os.getenv("API_TOKEN")
    HEADERS = {"Authorization": f"Bearer {API_KEY}"}
    API_URL = os.getenv("EMBED_API_URL")

    def __init__(self):
        pass

    def embed_documents(self,texts: List[str]) -> List[List[float]]:
        print(texts)
        rest_client = requests.Session()
        response = rest_client.post(
            self.API_URL,json={"inputs":texts},headers=self.HEADERS
        )
        return response.json()

    def embed_query(self,text: str) -> List[float]:
        print(f"embedding text {text}")
        response = self.embed_documents([text])[0]
        return response

"""
if __name__ == "__main__":
    custom_embeddings = CustomEmbeddings()
    sample_texts = ["Hello world!","hello"]
    embeddings = custom_embeddings.embed_documents(sample_texts)
    sample_text = "Hello world!"
    embeddings2 = custom_embeddings.embed_query(sample_text)
    print(embeddings2)
"""