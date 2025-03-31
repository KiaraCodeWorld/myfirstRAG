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


=====
