# Install required packages
# pip install pgvector langchain langchain-community openai tiktoken psycopg2-binary pypdf

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import PGVector
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF Loading and Processing
pdf_loader = PyPDFLoader("path/to/your/document.pdf")
raw_documents = pdf_loader.load()

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
documents = text_splitter.split_documents(raw_documents)

# PostgreSQL Vector Store Configuration
CONNECTION_STRING = "postgresql://user:password@localhost:5432/dbname"
COLLECTION_NAME = "pdf_hybrid_search"

# Initialize Embeddings and Store Documents
embeddings = OpenAIEmbeddings()
vector_store = PGVector.from_documents(
    embedding=embeddings,
    documents=documents,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

# Create Retrievers
vector_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# BM25 Retriever (in-memory)
texts = [doc.page_content for doc in documents]
bm25_retriever = BM25Retriever.from_texts(texts)
bm25_retriever.k = 5

# Hybrid Search Ensemble
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # Adjust based on your use case
)

# Query Execution
query = "Your search query here"
results = hybrid_retriever.invoke(query)

# Display results with sources
for doc in results:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}\n{'='*50}")

==========

import os
from langchain.vectorstores import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
import sqlalchemy
from sqlalchemy import create_engine, text
from tqdm import tqdm  # For progress bar
import glob

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # Replace with your OpenAI API key

# Database configuration
DATABASE_URL = "postgresql+psycopg2://username:password@localhost:5432/your_database"

# Initialize the database engine
engine = create_engine(DATABASE_URL)

# Ensure the pgvector extension is enabled
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

# Set up the embedding function and vector store
embedding_dimension = 1536  # Dimension for OpenAI's text-embedding-ada-002
collection_name = "documents_collection"

embeddings = OpenAIEmbeddings()

# Create the PGVector vector store
vectorstore = PGVector(
    connection_string=DATABASE_URL,
    embedding_function=embeddings,
    collection_name=collection_name,
    dimension=embedding_dimension,
    # Uncomment the following line if you want to start with a fresh collection
    # pre_delete_collection=True
)

# Function to load PDF documents from a directory
def load_pdfs_from_directory(directory_path):
    pdf_files = glob.glob(f"{directory_path}/*.pdf")
    documents = []
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        loader = PyPDFLoader(pdf_file)
        pdf_documents = loader.load_and_split()
        documents.extend(pdf_documents)
    return documents

# Load PDF documents from the specified directory
directory_of_pdfs = "/path/to/your/pdf/directory"  # Replace with your directory path
docs = load_pdfs_from_directory(directory_of_pdfs)

# Add documents to the vector store
vectorstore.add_documents(docs)

# Function to perform keyword search using PostgreSQL full-text search
def keyword_search_pg(query):
    with engine.connect() as conn:
        result = conn.execute(
            text(f"""
            SELECT document_id, document_content
            FROM {collection_name}
            WHERE to_tsvector('english', document_content) @@ plainto_tsquery('english', :query)
            """),
            {"query": query}
        )
        return [dict(row) for row in result]

# Function to perform hybrid search combining keyword and semantic search
def hybrid_search(query, top_k=5, keyword_weight=0.5, semantic_weight=0.5):
    # Perform keyword search
    keyword_results = keyword_search_pg(query)

    # Perform semantic search
    semantic_results = vectorstore.similarity_search_with_score(query, k=top_k)

    # Map contents to scores
    semantic_scores = {}
    max_score = max(score for _, score in semantic_results) or 1e-6  # Avoid division by zero
    for doc, score in semantic_results:
        # Normalize the score (lower distance means more similar)
        normalized_score = 1 - (score / max_score)
        semantic_scores[doc.page_content] = normalized_score

    # Map keyword results
    keyword_scores = {doc['document_content']: 1.0 for doc in keyword_results}

    # Combine scores
    combined_scores = {}
    all_contents = set(keyword_scores.keys()).union(semantic_scores.keys())
    for content in all_contents:
        combined_scores[content] = (
            keyword_scores.get(content, 0) * keyword_weight +
            semantic_scores.get(content, 0) * semantic_weight
        )

    # Get documents and sort by combined score
    combined_docs = [{'content': content, 'score': combined_scores[content]} for content in combined_scores]
    combined_docs.sort(key=lambda x: -x['score'])

    return combined_docs[:top_k]

# Example usage of hybrid search
query = "machine learning algorithms"
results = hybrid_search(query, top_k=5)

print("\nHybrid Search Results:")
for idx, r in enumerate(results, start=1):
    print(f"\nResult {idx}:")
    print(f"Content (truncated): {r['content'][:200]}...")  # Truncate for display
    print(f"Combined Score: {r['score']:.4f}")

============

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain.retrievers.ensemble import EnsembleRetriever
import psycopg2
from psycopg2 import sql
from typing import List

# Load environment variables
load_dotenv()

class HybridSearch:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.conn = psycopg2.connect(os.getenv("POSTGRES_CONNECTION_STRING"))
        
        # Initialize database schema
        self._init_db()

    def _init_db(self):
        """Initialize database tables and indexes"""
        with self.conn.cursor() as cur:
            # Create documents table with both vector and full-text search capabilities
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    embedding VECTOR(1536),
                    search_vector TSVECTOR
                )
            """)
            
            # Create indexes
            cur.execute("CREATE INDEX IF NOT EXISTS embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops)")
            cur.execute("CREATE INDEX IF NOT EXISTS search_vector_idx ON documents USING GIN (search_vector)")
            
            self.conn.commit()

    def load_documents(self, file_path: str):
        """Load and process PDF documents"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        
        # Store chunks in database
        with self.conn.cursor() as cur:
            for chunk in chunks:
                content = chunk.page_content
                
                # Generate embedding
                embedding = self.embeddings.embed_query(content)
                
                # Insert document with both vector and full-text search data
                cur.execute(
                    sql.SQL("""
                        INSERT INTO documents (content, embedding, search_vector)
                        VALUES (%s, %s, to_tsvector('english', %s))
                    """),
                    (content, embedding, content)
                )
            self.conn.commit()

    class VectorRetriever(BaseRetriever):
        """Semantic search retriever using vector embeddings"""
        def __init__(self, conn, embeddings):
            super().__init__()
            self.conn = conn
            self.embeddings = embeddings

        def _get_relevant_documents(self, query: str) -> List[Document]:
            query_embedding = self.embeddings.embed_query(query)
            docs = []
            
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT id, content, 1 - (embedding <=> %s) AS score
                    FROM documents
                    ORDER BY embedding <=> %s
                    LIMIT 5
                """, (query_embedding, query_embedding))
                
                for row in cur.fetchall():
                    docs.append(Document(
                        page_content=row[1],
                        metadata={"score": row[2], "id": row[0]}
                    ))
            return docs

    class KeywordRetriever(BaseRetriever):
        """Keyword search retriever using full-text search"""
        def __init__(self, conn):
            super().__init__()
            self.conn = conn

        def _get_relevant_documents(self, query: str) -> List[Document]:
            docs = []
            
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT id, content, ts_rank(search_vector, plainto_tsquery('english', %s)) as score
                    FROM documents
                    WHERE search_vector @@ plainto_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT 5
                """, (query, query))
                
                for row in cur.fetchall():
                    docs.append(Document(
                        page_content=row[1],
                        metadata={"score": row[2], "id": row[0]}
                    ))
            return docs

    def get_hybrid_retriever(self):
        """Create ensemble retriever combining both search methods"""
        vector_retriever = self.VectorRetriever(self.conn, self.embeddings)
        keyword_retriever = self.KeywordRetriever(self.conn)
        
        return EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[0.4, 0.6]  # Adjust weights based on your use case
        )

# Example usage
if __name__ == "__main__":
    hs = HybridSearch()
    
    # Load PDF documents (only needs to be done once)
    # hs.load_documents("your-document.pdf")
    
    # Get hybrid retriever
    retriever = hs.get_hybrid_retriever()
    
    # Perform hybrid search
    query = "Your search query here"
    results = retriever.get_relevant_documents(query)
    
    # Display results
    for doc in results:
        print(f"Score: {doc.metadata['score']:.3f}")
        print(doc.page_content)
        print("\n" + "-"*50 + "\n")
