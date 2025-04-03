from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import BaseDocumentCompressor
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import numpy as np

class BgeSimilarityReranker(BaseDocumentCompressor):
    def __init__(self, embeddings, top_n=5):
        self.embeddings = embeddings
        self.top_n = top_n

    def compress_documents(self, documents, query, **kwargs):
        # Generate embeddings
        query_embed = self.embeddings.embed_query(query)
        doc_embeds = self.embeddings.embed_documents([d.page_content for d in documents])
        
        # Calculate cosine similarities
        similarities = []
        for doc_embed in doc_embeds:
            norm = np.linalg.norm
            cos_sim = np.dot(query_embed, doc_embed) / (norm(query_embed)*norm(doc_embed))
            similarities.append(cos_sim)
        
        # Sort and select top documents
        sorted_docs = [doc for _, doc in sorted(zip(similarities, documents), 
                                              key=lambda x: x[0], 
                                              reverse=True)]
        return sorted_docs[:self.top_n]

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
reranker = BgeSimilarityReranker(embeddings, top_n=5)

# Set up retrieval pipeline with expanded initial results
base_retriever = your_vectorstore.as_retriever(search_kwargs={"k": 20})  # Retrieve 20 docs initially
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)

# Execute search
compressed_docs = compression_retriever.invoke("Economic policy changes?")

==========

# First install required packages
# pip install langchain sentence-transformers numpy

from langchain.retrievers import BM25Retriever
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np

# Sample documents
documents = [
    "Cats are domestic animals.",
    "Penguins are flightless birds found in Antarctica.",
    "Python is a popular programming language.",
    "The Eiffel Tower is in Paris.",
    "Mars is called the Red Planet.",
]

# Initialize BM25 retriever for initial retrieval
bm25_retriever = BM25Retriever.from_texts(documents)
bm25_retriever.k = 5  # Get all documents for demonstration

# Your query
query = "Birds that cannot fly"

# Initial document retrieval
initial_results = bm25_retriever.invoke(query)
print("Initial BM25 results:")
for doc in initial_results:
    print(f"- {doc.page_content}")

# Initialize embedding model (BGE or vlm-to-vec)
def get_embedder(model_name="BAAI/bge-base-en"):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={
            'normalize_embeddings': True  # Important for cosine similarity
        }
    )

# Choose model (swap these to test different models)
embedder = get_embedder("BAAI/bge-base-en")  # BGE model
# embedder = get_embedder("vikhyatk/vlm-to-vec")  # vlm-to-vec model

# Generate embeddings
query_embedding = embedder.embed_query(query)
doc_embeddings = embedder.embed_documents([doc.page_content for doc in initial_results])

# Calculate cosine similarities
similarities = np.dot(doc_embeddings, query_embedding)

# Combine documents with their scores
scored_docs = list(zip(initial_results, similarities))

# Sort documents by similarity score
reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

# Display final results
print("\nReranked results:")
for doc, score in reranked_docs:
    print(f"- [{score:.3f}] {doc.page_content}")

=========

# First install required packages
# pip install langchain sentence-transformers numpy

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever
import numpy as np

# Sample documents
documents = [
    "Penguins are flightless birds living in cold climates",
    "Bats are the only mammals capable of sustained flight",
    "The history of aviation shows human flight development",
    "Ostriches are large flightless birds from Africa",
    "Airplanes use wings to generate lift for flight"
]

# Initialize BM25 retriever for first-stage retrieval
bm25_retriever = BM25Retriever.from_texts(documents)
bm25_retriever.k = 5  # Retrieve all documents for demonstration

# Initialize BGE embedder
embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' for GPU
    encode_kwargs={'normalize_embeddings': True}  # Crucial for cosine similarity
)

def rerank_documents(query, documents):
    # Generate embeddings
    query_embed = embedder.embed_query(query)
    doc_embeds = embedder.embed_documents([d.page_content for d in documents])
    
    # Calculate cosine similarities
    similarities = np.dot(doc_embeds, query_embed)
    
    # Pair documents with scores and sort
    scored_docs = sorted(zip(documents, similarities), 
                        key=lambda x: x[1], 
                        reverse=True)
    return scored_docs

# Search query
query = "Find me information about birds that cannot fly"

# First-stage retrieval
initial_docs = bm25_retriever.invoke(query)
print("Initial BM25 Results:")
for doc in initial_docs:
    print(f"- {doc.page_content}")

# Rerank using BGE embeddings
reranked_results = rerank_documents(query, initial_docs)

print("\nReranked Results:")
for doc, score in reranked_results:
    print(f"- [Score: {score:.3f}] {doc.page_content}")

=========

O1 - preview : 

import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Sample documents
documents = [
    "The cat sat on the mat.",
    "Dogs are man's best friend.",
    "The sun rises in the east.",
    "Python is a popular programming language.",
    "Artificial Intelligence is the future."
]

# Convert documents to LangChain Document objects
docs = [Document(page_content=doc) for doc in documents]

# Your query
query = "Tell me about programming languages"

# Initialize the BGE embedding model from Hugging Face
embedding_model_name = "BAAI/bge-small-en"  # You can also use 'BAAI/bge-base-en' or 'BAAI/bge-large-en'
hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Embed the documents
doc_embeddings = hf_embeddings.embed_documents([doc.page_content for doc in docs])

# Embed the query
query_embedding = hf_embeddings.embed_query(query)

# Compute cosine similarity between query and each document
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity_scores = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in doc_embeddings]

# Rerank the documents based on similarity scores
reranked_docs = sorted(zip(similarity_scores, docs), key=lambda x: x[0], reverse=True)

# Print the reranked documents
print("Reranked Documents:")
for score, doc in reranked_docs:
    print(f"Score: {score:.4f} - Document: {doc.page_content}")
