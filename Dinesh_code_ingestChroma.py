import os
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import uuid
import csv


def load_file(filename):
    client = chromadb.PersistentClient(path="./db")
    collection = client.get_or_create_collection(name="vector_db")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(embeddings)

    if filename.endswith('.pdf'):
        loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
        # loader = DirectoryLoader('data/',filename=filename, show_progress=True, loader_cls=PyPDFLoader)

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        texts = text_splitter.split_documents(documents)

        persist_directory = "db"
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name="vector_db"
        )
        vectordb.persist()

    if filename.endswith('.csv'):
        def get_csv_file(filename):
            # Read the data from the CSV file
            with open(filename, "r") as f:
                # Skip the header row
                next(f)
                reader = csv.reader(f)
                return list(reader)

        data = get_csv_file(filename)

        # Flatten the data into two lists
        ids = [str(uuid.uuid1()) for arr in data]
        docs = [arr[1] for arr in data]

        # Split the data into chunks
        chunk_size = 1000

        id_chunks = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
        doc_chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]

        for id_chunk, doc_chunk in zip(id_chunks, doc_chunks):
            collection.add(
                ids=id_chunk,
                documents=doc_chunk  # ,
                # metadatas=[{"city":"category"} for _ in id_chunk]
            )


load_file("ContactFB.pdf")
print("Chroma DB Successfully Created!")