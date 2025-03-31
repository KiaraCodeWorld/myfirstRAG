from langchain_postgres.vectorstores import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Configuration
os.environ["PGVECTOR_CONNECTION_STRING"] = "postgresql+psycopg://user:password@localhost:5432/dbname"

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load and split PDF
loader = PyPDFLoader("your_document.pdf")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
pages = loader.load_and_split(text_splitter)

# Add metadata for hybrid search
for i, doc in enumerate(pages):
    doc.metadata.update({
        "page_number": i+1,
        "source": f"pdf_page_{i+1}",
        "text": doc.page_content  # Store raw text for BM25
    })

# Create vector store with hybrid search support
vector_store = PGVector.from_documents(
    documents=pages,
    embedding=embeddings,
    collection_name="pdf_collection",
    use_jsonb=True,
    pre_delete_collection=True,
    distance_strategy="COSINE"
)


###

from langchain_postgres.vectorstores import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Configuration
os.environ["PGVECTOR_CONNECTION_STRING"] = "postgresql+psycopg://user:password@localhost:5432/dbname"

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load and split PDF
loader = PyPDFLoader("your_document.pdf")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
pages = loader.load_and_split(text_splitter)

# Add metadata for hybrid search
for i, doc in enumerate(pages):
    doc.metadata.update({
        "page_number": i+1,
        "source": f"pdf_page_{i+1}",
        "text": doc.page_content  # Store raw text for BM25
    })

# Create vector store with hybrid search support
vector_store = PGVector.from_documents(
    documents=pages,
    embedding=embeddings,
    collection_name="pdf_collection",
    use_jsonb=True,
    pre_delete_collection=True,
    distance_strategy="COSINE"
)


-----

from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Initialize retrievers
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
bm25_retriever = BM25Retriever.from_documents(pages)

# Create hybrid retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

# Execute hybrid search
results = hybrid_retriever.invoke("your search query")

-------

from pdfminer.high_level import extract_pages

def extract_links(page):
    # Implement custom link extraction logic
    return []

for i, (page_content, layout) in enumerate(zip(pages, extract_pages("your_document.pdf"))):
    pages[i].metadata["links"] = extract_links(layout)


======= deepseek

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.schema import Document
import psycopg2
from sqlalchemy import text

# Step 1: Load and split PDF document
loader = PyPDFLoader("your_document.pdf")
pages = loader.load_and_split()

# Step 2: Prepare documents with metadata
docs = []
for page_num, page in enumerate(pages, start=1):
    doc = Document(
        page_content=page.page_content,
        metadata={
            "page_number": page_num,
            "page_link": f"your_document.pdf#page={page_num}",
            "text": page.page_content  # Store raw text separately if needed
        }
    )
    docs.append(doc)

# Step 3: Create PostgreSQL connection string
CONNECTION_STRING = "postgresql://username:password@localhost:5432/vectordb"
COLLECTION_NAME = "pdf_documents"

# Step 4: Create vector store
embeddings = OpenAIEmbeddings()

# Store documents in PGVector
db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

# Step 5: Create hybrid search indexes (run once)
def create_hybrid_indexes():
    conn = psycopg2.connect(CONNECTION_STRING)
    cursor = conn.cursor()
    
    # Create full-text search index
    cursor.execute(text("""
        ALTER TABLE langchain_pg_embedding
        ADD COLUMN IF NOT EXISTS text_search_vector tsvector;
    """))
    
    cursor.execute(text("""
        UPDATE langchain_pg_embedding
        SET text_search_vector = to_tsvector('english', metadata->>'text');
    """))
    
    cursor.execute(text("""
        CREATE INDEX IF NOT EXISTS text_search_idx 
        ON langchain_pg_embedding 
        USING GIN(text_search_vector);
    """))
    
    conn.commit()
    cursor.close()
    conn.close()

create_hybrid_indexes()

# Step 6: Example hybrid search function
def hybrid_search(query, top_k=5):
    # Generate embedding for the query
    query_embedding = embeddings.embed_query(query)
    
    # Convert to PostgreSQL vector string
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    
    # Connect to database
    conn = psycopg2.connect(CONNECTION_STRING)
    cursor = conn.cursor()
    
    # Hybrid search query combining vector and full-text search
    cursor.execute(text("""
        SELECT 
            metadata->>'page_number' as page_number,
            metadata->>'page_link' as page_link,
            metadata->>'text' as text,
            (0.5 * (1 - embedding <=> :embedding)) + 
            (0.5 * ts_rank(text_search_vector, plainto_tsquery('english', :query))) 
            AS combined_score
        FROM langchain_pg_embedding
        WHERE 
            text_search_vector @@ plainto_tsquery('english', :query)
            OR embedding <=> :embedding < 0.85
        ORDER BY combined_score DESC
        LIMIT :top_k;
    """), {'embedding': embedding_str, 'query': query, 'top_k': top_k})
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return [{
        "page_number": int(r[0]),
        "page_link": r[1],
        "text": r[2],
        "score": float(r[3])
    } for r in results]

# Example usage
results = hybrid_search("search query here")
for result in results:
    print(f"Page {result['page_number']} ({result['page_link']}) - Score: {result['score']:.4f}")
    print(f"Text: {result['text'][:200]}...\n")
