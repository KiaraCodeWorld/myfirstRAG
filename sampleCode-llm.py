# First install required packages
# pip install chromadb langchain sentence-transformers

import chromadb
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings

# Native Chroma DB approach
def native_chroma_example():
    client = chromadb.Client()
    collection = client.create_collection("native_example")
    
    documents = [
        "The capital of France is Paris",
        "Python was created by Guido van Rossum",
        "Water boils at 100°C at sea level",
        "Blockchain is a decentralized ledger technology",
        "Machine learning uses statistical methods"
    ]
    
    ids = ["doc001", "doc002", "doc003", "doc004", "doc005"]
    
    collection.add(documents=documents, ids=ids)
    results = collection.get()
    print("Native Chroma Results:")
    print(results)

# LangChain Chroma.from_documents approach
def langchain_chroma_example():
    documents = [
        Document(
            page_content="Mars is the fourth planet from the Sun",
            metadata={"source": "astronomy"}
        ),
        Document(
            page_content="GDP measures a country's economic output",
            metadata={"source": "economics"}
        ),
        Document(
            page_content="Shakespeare wrote Hamlet and Macbeth",
            metadata={"source": "literature"}
        ),
        Document(
            page_content="Photosynthesis converts light to chemical energy",
            metadata={"source": "biology"}
        ),
        Document(
            page_content="HTTP is the foundation of data communication for the Web",
            metadata={"source": "computer_science"}
        )
    ]
    
    ids = ["doc101", "doc102", "doc103", "doc104", "doc105"]
    
    # Create vector store with embeddings
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        ids=ids,
        persist_directory="./chroma_db"
    )
    
    # Query example
    results = vector_store.similarity_search("science topics", k=2)
    print("\nLangChain Similarity Search Results:")
    for doc in results:
        print(f"\n{doc.page_content}\nMetadata: {doc.metadata}")

# Run both examples
native_chroma_example()
langchain_chroma_example()
