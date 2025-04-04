
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

import numpy as np
from typing import List, Dict, Any, Tuple

# Custom reranker using embedding similarity
class EmbeddingReranker:
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
    
    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """
        Rerank documents based on similarity between query embedding and document embeddings
        """
        if not documents:
            return []
        
        # Get query embedding
        query_embedding = self.embeddings_model.embed_query(query)
        
        # Get document embeddings and calculate similarity
        doc_embeddings = []
        for doc in documents:
            # If document already has embedding, use it; otherwise compute it
            if hasattr(doc, 'embedding') and doc.embedding is not None:
                doc_embeddings.append(doc.embedding)
            else:
                # Assuming document.page_content contains the text to embed
                doc_embedding = self.embeddings_model.embed_documents([doc.page_content])[0]
                doc.embedding = doc_embedding
                doc_embeddings.append(doc_embedding)
        
        # Calculate cosine similarity between query and documents
        similarities = []
        for doc_embedding in doc_embeddings:
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        # Create pairs of (document, similarity)
        doc_similarity_pairs = list(zip(documents, similarities))
        
        # Sort by similarity (descending)
        ranked_docs = [doc for doc, _ in sorted(doc_similarity_pairs, 
                                                key=lambda x: x[1], 
                                                reverse=True)]
        
        # Return top N documents
        return ranked_docs[:top_n]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        # Convert to numpy arrays for efficient calculation
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
        
        return dot_product / (norm_vec1 * norm_vec2)

# Example usage
def main():
    # Initialize embedding model
    embeddings = OpenAIEmbeddings()
    
    # Sample documents
    documents = [
        Document(page_content="Machine learning models require large datasets", 
                 metadata={"source": "textbook", "page": 15}),
        Document(page_content="Neural networks are a type of machine learning model",
                 metadata={"source": "article", "page": 2}),
        Document(page_content="Python is a popular programming language for data science",
                 metadata={"source": "tutorial", "page": 7}),
        Document(page_content="Transformers have revolutionized natural language processing",
                 metadata={"source": "paper", "page": 1}),
        Document(page_content="Data preprocessing is crucial for machine learning success",
                 metadata={"source": "guide", "page": 23}),
    ]
    
    # Create a vector store
    vectorstore = Chroma.from_documents(documents, embeddings)
    
    # Basic retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Create our custom reranker
    reranker = EmbeddingReranker(embeddings)
    
    # Use as a standalone reranker
    query = "machine learning techniques"
    initial_results = retriever.get_relevant_documents(query)
    print("Initial results:")
    for i, doc in enumerate(initial_results):
        print(f"{i+1}. {doc.page_content}")
    
    reranked_results = reranker.rerank(query, initial_results, top_n=3)
    print("\nReranked results:")
    for i, doc in enumerate(reranked_results):
        print(f"{i+1}. {doc.page_content}")
    
    # Alternative: Use with LangChain's ContextualCompressionRetriever
    # This uses EmbeddingsFilter which is LangChain's built-in embedding-based reranker
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, 
                                        similarity_threshold=0.7)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=retriever
    )
    
    compressed_docs = compression_retriever.get_relevant_documents(query)
    print("\nResults from ContextualCompressionRetriever:")

  ====   Perplexity : 

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import BaseDocumentCompressor
from langchain_core.documents import Document
from typing import List
from langchain_openai import OpenAIEmbeddings
import numpy as np

class EmbeddingSimilarityReranker(BaseDocumentCompressor):
    """Custom reranker using embedding similarity scores"""
    
    def __init__(self, embedding_model, top_n=3):
        self.embedding_model = embedding_model
        self.top_n = top_n
        
    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Calculate similarities
        for doc in documents:
            doc_embedding = self.embedding_model.embed_query(doc.page_content)
            similarity = np.dot(query_embedding, doc_embedding)
            doc.metadata["similarity_score"] = similarity
            
        # Sort by similarity score and return top_n
        sorted_docs = sorted(documents, 
                           key=lambda x: -x.metadata["similarity_score"])
        return sorted_docs[:self.top_n]

# Initialize components
embedding_model = OpenAIEmbeddings()
retriever = ...  # Your base retriever (e.g., vector store retriever)

# Create custom reranker
reranker = EmbeddingSimilarityReranker(embedding_model, top_n=5)

# Build compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever
)

# Example usage
compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown?"
)

=========== 
    for i, doc in enumerate(compressed_docs):
        print(f"{i+1}. {doc.page_content}")

if __name__ == "__main__":
    main()

========== mistral 

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BaseRetriever
from langchain.rerankers import EmbeddingReranker

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the retriever (assuming you have a retriever implemented)
retriever = BaseRetriever()

# Retrieve documents
documents = retriever.retrieve(query="Your query here")

# Initialize the reranker with the embedding model
reranker = EmbeddingReranker(embedding_model=embedding_model)

# Rerank the documents
reranked_documents = reranker.rerank(documents, query="Your query here")

# Print the reranked documents
for doc in reranked_documents:
    print(doc)

=====  Copilot : 

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import numpy as np

# Define embedding model
embedding_model = OpenAIEmbeddings()  # Replace with your preferred embedding model

# Example candidate documents
documents = [
    {"text": "LangChain enables seamless AI integrations."},
    {"text": "FAISS is an efficient similarity search library."},
    {"text": "Embedding models produce vector representations of text."},
]

# Query to rerank based on relevance
query = "How does LangChain work with embeddings?"

# Generate embeddings for the query and documents
query_embedding = embedding_model.embed_query(query)
doc_embeddings = [embedding_model.embed_query(doc["text"]) for doc in documents]

# Compute cosine similarity scores
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

similarities = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in doc_embeddings]

# Rerank documents based on similarity scores
ranked_documents = sorted(
    zip(documents, similarities),
    key=lambda x: x[1],
    reverse=True
)

# Output reranked documents
print("Reranked Documents:")
for doc, score in ranked_documents:
    print(f"Score: {score:.4f}, Text: {doc['text']}")
