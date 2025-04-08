=== deepseek 

from typing import List, Union
import requests
from langchain.schema.embeddings import Embeddings
from abc import ABC, abstractmethod

class CustomAPIEmbedder(Embeddings):
    """Custom embedding model using external API"""
    
    def __init__(
        self,
        api_url: str = "https://api.example.com/embeddings",
        api_token: str = "your_api_token",
        timeout: int = 30,
        model_name: str = "company-embedding-model-v1"
    ):
        self.api_url = api_url
        self.api_token = api_token
        self.timeout = timeout
        self.model_name = model_name

    def _call_api(self, text: str) -> List[float]:
        """Helper method to call the embedding API"""
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": text,
            "model": self.model_name
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()['data'][0]['embedding']
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API call failed: {str(e)}")
        except KeyError:
            raise ValueError("Unexpected API response format")

    def embed_query(self, text: str) -> List[float]:
        """Embed single text query"""
        return self._call_api(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return [self._call_api(text) for text in texts]

# Usage example
if __name__ == "__main__":
    # Initialize with your API credentials
    embedder = CustomAPIEmbedder(
        api_url="YOUR_API_ENDPOINT",
        api_token="YOUR_API_KEY",
        model_name="your-model-name"
    )
    
    # Test single query
    query_embedding = embedder.embed_query("Hello world")
    print(f"Query embedding (first 5 dims): {query_embedding[:5]}")
    print(f"Embedding dimension: {len(query_embedding)}")
    
    # Test multiple documents
    docs = ["First document", "Second document"]
    doc_embeddings = embedder.embed_documents(docs)
    print(f"Number of document embeddings: {len(doc_embeddings)}")
    print(f"First doc embedding dimension: {len(doc_embeddings[0])}")

======= claude 

from langchain_core.embeddings import Embeddings
from typing import List
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class CustomAPIEmbedder(Embeddings):
    """Custom embedder that calls an external API for embeddings."""
    
    def __init__(
        self, 
        api_url: str = None,
        api_key: str = None,
        model_name: str = "text-embedding-ada-002",
        dimensions: int = 1536,
        batch_size: int = 8
    ):
        """Initialize the custom embedder.
        
        Args:
            api_url: URL of the embedding API endpoint.
            api_key: API key for authentication.
            model_name: Name of the embedding model to use.
            dimensions: Dimensionality of the embedding vectors.
            batch_size: Number of texts to embed in a single API call.
        """
        self.api_url = api_url or os.getenv("EMBEDDING_API_URL")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY")
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size
        
        if not self.api_url:
            raise ValueError("API URL must be provided or set as EMBEDDING_API_URL environment variable")
        if not self.api_key:
            raise ValueError("API key must be provided or set as EMBEDDING_API_KEY environment variable")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.
        
        Args:
            texts: List of documents to embed.
            
        Returns:
            List of embedding vectors, one for each document.
        """
        all_embeddings = []
        
        # Process in batches to avoid overloading the API
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            
            # Make the API request
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "input": batch
                }
            )
            
            # Handle the API response
            if response.status_code != 200:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
            
            result = response.json()
            batch_embeddings = [item["embedding"] for item in result["data"]]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.
        
        Args:
            text: Query text to embed.
            
        Returns:
            Embedding vector for the query.
        """
        # Reuse the document embedding method for consistency
        embeddings = self.embed_documents([text])
        return embeddings[0]


# Example usage
if __name__ == "__main__":
    # Create the custom embedder
    embedder = CustomAPIEmbedder()
    
    # Example documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models transform input data into useful predictions",
        "Natural language processing helps computers understand human language"
    ]
    
    # Get embeddings for documents
    doc_embeddings = embedder.embed_documents(documents)
    print(f"Document embeddings shape: {len(doc_embeddings)} documents, {len(doc_embeddings[0])} dimensions each")
    
    # Get embedding for a query
    query = "How do AI systems process text data?"
    query_embedding = embedder.embed_query(query)
    print(f"Query embedding shape: {len(query_embedding)} dimensions")
    
    # You can now use these embeddings for semantic search, clustering, etc.

===== perplexity 

import requests
from typing import List
from langchain.embeddings import Embeddings

class CustomAPIEmbeddings(Embeddings):
    """
    Custom embedding class to interact with an API-based embedding model.
    """

    def __init__(self, model_name: str, api_url: str):
        """
        Initialize the custom embedder.

        Args:
            model_name (str): Name of the embedding model.
            api_url (str): URL of the embedding API endpoint.
        """
        self.model_name = model_name
        self.api_url = api_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents by sending them to the API.

        Args:
            texts (List[str]): List of document texts to embed.

        Returns:
            List[List[float]]: List of embeddings for the documents.
        """
        response = requests.post(
            self.api_url,
            json={"model_name": self.model_name, "texts": texts}
        )
        
        # Ensure response handling matches your API's output format
        if response.status_code == 200:
            return response.json()["embeddings"]
        else:
            raise Exception(f"Failed to fetch embeddings: {response.text}")

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query by sending it to the API.

        Args:
            text (str): Query text to embed.

        Returns:
            List[float]: Embedding for the query.
        """
        return self.embed_documents([text])[0]

# Example usage
if __name__ == "__main__":
    # Replace with your actual model name and API endpoint
    model_name = "example-model"
    api_url = "https://api.example.com/embed"

    # Instantiate the custom embedder
    custom_embedder = CustomAPIEmbeddings(model_name=model_name, api_url=api_url)

    # Test embedding methods
    query_embedding = custom_embedder.embed_query("What is LangChain?")
    print(f"Query Embedding Dimension: {len(query_embedding)}")

    document_embeddings = custom_embedder.embed_documents(["Document 1", "Document 2"])
    print(f"Document Embedding Dimensions: {[len(doc) for doc in document_embeddings]}")

=== galaxy - o1 preview

from typing import List
import requests
from langchain.embeddings.base import Embeddings

class CustomAPIEmbeddings(Embeddings):
    def __init__(self, api_url: str):
        """Initialize with the API URL of the embedding service."""
        self.api_url = api_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using the external embedding API.
        
        Args:
            texts (List[str]): A list of texts to embed.
        
        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        # Prepare the payload for the API request
        payload = {"texts": texts}

        # Call the embedding API
        response = requests.post(self.api_url, json=payload)

        # Check for a successful response
        if response.status_code != 200:
            raise Exception(
                f"API request failed with status code {response.status_code}: {response.text}"
            )

        # Extract embeddings from the response
        embeddings = response.json().get("embeddings", [])

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text using the external embedding API.
        
        Args:
            text (str): The text to embed.
        
        Returns:
            List[float]: The embedding vector.
        """
        embeddings = self.embed_documents([text])
        return embeddings[0]

# Example usage
if __name__ == "__main__":
    api_url = "https://your-embedding-api.com/embeddings"  # Replace with your API URL
    embedder = CustomAPIEmbeddings(api_url)

    # Embedding multiple documents
    texts = ["Hello, world!", "LangChain makes building applications easier."]
    embeddings = embedder.embed_documents(texts)
    print("Document Embeddings:")
    for text, embedding in zip(texts, embeddings):
        print(f"Text: {text}\nEmbedding: {embedding}\n")

    # Embedding a single query
    query = "What is LangChain?"
    query_embedding = embedder.embed_query(query)
    print("Query Embedding:")
    print(f"Query: {query}\nEmbedding: {query_embedding}")

=== grok2 

from langchain.embeddings import Embeddings
import requests

class CustomAPIEmbedder(Embeddings):
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"text": text}
            )
            if response.status_code == 200:
                embedding = response.json()['embedding']
                embeddings.append(embedding)
            else:
                raise ValueError(f"API request failed with status code {response.status_code}")
        return embeddings

    def embed_query(self, text):
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"text": text}
        )
        if response.status_code == 200:
            return response.json()['embedding']
        else:
            raise ValueError(f"API request failed with status code {response.status_code}")

# Usage example
api_url = "https://your-api-endpoint.com/embed"
api_key = "your_api_key_here"

embedder = CustomAPIEmbedder(api_url, api_key)

# Embedding a document
documents = ["This is a sample document.", "Another document for testing."]
document_embeddings = embedder.embed_documents(documents)
print("Document embeddings:", document_embeddings)

# Embedding a query
query = "What is the purpose of this document?"
query_embedding = embedder.embed_query(query)
print("Query embedding:", query_embedding)


