import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import uuid
import os

# Setup ChromaDB client
def setup_chroma_client(persist_directory="./chroma_db"):
    """Set up and return a ChromaDB client with persistence."""
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(
            anonymized_telemetry=False
        )
    )
    return client

# Create a collection with OpenAI embeddings
def create_collection(client, collection_name="documents_collection"):
    """Create and return a collection with OpenAI embeddings."""
    # You'll need OpenAI API key set as env variable
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-ada-002"
    )
    
    # For demo purposes, you can also use a basic embedding function
    # all_minilm_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Create or get collection
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists. Using existing collection.")
    except:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=openai_ef,
            metadata={"description": "Collection of document embeddings"}
        )
        print(f"Created new collection '{collection_name}'.")
    
    return collection

# Sample documents
def get_sample_documents():
    """Return a list of 5 sample documents with their IDs."""
    documents = [
        "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.",
        "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability.",
        "Quantum computing is the exploitation of collective properties of quantum states, such as superposition and entanglement.",
        "Climate change includes both global warming driven by human-induced emissions of greenhouse gases.",
        "Artificial intelligence is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans."
    ]
    
    # Generate document IDs
    doc_ids = [str(uuid.uuid4())[:8] for _ in range(len(documents))]
    
    # Create metadata for each document
    metadata = [
        {"source": "textbook", "topic": "machine learning", "year": 2023},
        {"source": "documentation", "topic": "programming", "year": 2022},
        {"source": "research paper", "topic": "physics", "year": 2024},
        {"source": "report", "topic": "environment", "year": 2023},
        {"source": "article", "topic": "technology", "year": 2024}
    ]
    
    return documents, doc_ids, metadata

# Add documents to collection
def add_documents_to_collection(collection, documents, doc_ids, metadata):
    """Add documents to the collection."""
    collection.add(
        documents=documents,
        ids=doc_ids,
        metadatas=metadata
    )
    print(f"Added {len(documents)} documents to the collection.")
    return collection

# Query the collection
def query_collection(collection, query_text, n_results=3):
    """Query the collection and return results."""
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results

# Example of ChromaDB from_documents using LangChain integration
def chroma_from_documents_example():
    """Example using LangChain's Document class with ChromaDB."""
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.schema import Document
    
    # Sample documents as LangChain Document objects
    documents, doc_ids, metadata = get_sample_documents()
    langchain_docs = [
        Document(page_content=doc, metadata=meta) 
        for doc, meta in zip(documents, metadata)
    ]
    
    # Initialize the embedding function
    embeddings = OpenAIEmbeddings()
    
    # Create a Chroma vector store from documents
    vectorstore = Chroma.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        persist_directory="./langchain_chroma_db",
        collection_name="langchain_documents"
    )
    
    # Example of similarity search
    results = vectorstore.similarity_search(
        query="Tell me about machine learning", 
        k=2
    )
    
    print("\n--- LangChain Chroma Results ---")
    for doc in results:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}\n")
    
    return vectorstore

# Main execution
def main():
    print("Setting up ChromaDB...")
    client = setup_chroma_client()
    collection = create_collection(client)
    
    print("\nGenerating sample documents...")
    documents, doc_ids, metadata = get_sample_documents()
    
    # Print sample documents and their IDs
    print("\n--- Sample Documents and IDs ---")
    for i, (doc, doc_id) in enumerate(zip(documents, doc_ids)):
        print(f"ID: {doc_id}")
        print(f"Document: {doc}")
        print(f"Metadata: {metadata[i]}\n")
    
    print("Adding documents to ChromaDB collection...")
    collection = add_documents_to_collection(collection, documents, doc_ids, metadata)
    
    print("\nQuerying the collection...")
    query = "machine learning and artificial intelligence"
    results = query_collection(collection, query)
    
    print("\n--- Query Results ---")
    print(f"Query: '{query}'")
    for i, (doc, id, distance) in enumerate(zip(
        results['documents'][0], 
        results['ids'][0],
        results['distances'][0]
    )):
        print(f"Result {i+1}:")
        print(f"ID: {id}")
        print(f"Document: {doc}")
        print(f"Distance: {distance}\n")
    
    print("\nDemonstrating Chroma.from_documents with LangChain...")
    vectorstore = chroma_from_documents_example()
    
    print("\nAll operations completed successfully!")

if __name__ == "__main__":
    main()
