import os
import requests
import PyPDF2
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings
)
from langchain.embeddings import HuggingFaceEmbeddings
from embedder_smallBGE import CustomEmbedder


def pdf_to_text(file_path):
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range( len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    pdf_file.close()
    return text

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

print("*********************")

text = pdf_to_text("./data/pet.pdf")
chunks = text_splitter.split_text(text)
customer_embedder = CustomEmbedder()

# Convert chunks to vector representations and store in Chroma DB
documents_list = []
embeddings_list = []
ids_list = []


for i, chunk in enumerate(chunks):
    vector = customer_embedder(chunk) #embeddings.embed_query(chunk)

    documents_list.append(chunk)
    embeddings_list.append(vector)
    ids_list.append(f"petpdf_{i}")

#client = chromadb.PersistentClient(path="./db")
chroma_client = chromadb.PersistentClient(path="../mydb_smallBGE")
chroma_client.delete_collection("rag_collection_smallBGE")
collection = chroma_client.create_collection(name="rag_collection_smallBGE")

print(embeddings_list)

"""
collection.add(
            embeddings=embeddings_list,
            documents=documents_list,
            ids=ids_list
        )

print(collection.count())

print("*********************")
"""

"""
DATA_PATH = 'data/'
loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(texts)

API_TOKEN = "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"  # os.environ["API_TOKEN"]
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    #"https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"  # "https://c9ejquzh6yum3xqf.us-east-1.aws.endpoints.huggingface.cloud/"

class CustomEmbedder(EmbeddingFunction):
    def __init__(self) -> None:
        self.API_TOKEN = "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" #os.environ["API_TOKEN"]
        self.API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5" #"https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5" #"https://api-inference.huggingface.co/models/BAAI/bge-large-en"
        self.headers = {"Authorization": "Bearer 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"}

    def __call__(self, input: Documents) -> Embeddings:
        rest_client = requests.Session()
        response = rest_client.post(
            #self.API_URL, json={"inputs": input}, headers={"Authorization": f"Bearer {self.API_TOKEN}"}
            self.API_URL, headers=self.headers, json=input
        ).json()
        return response

documents_list = []
embeddings_list = []
ids_list = []

custom_embedder = CustomEmbedder()
client = chromadb.PersistentClient(path="./db")
chroma_client = chromadb.PersistentClient(path="../mydb")
#chroma_client.reset()
chroma_client.heartbeat()
chroma_client.delete_collection("rag_collection")
collection = chroma_client.create_collection(name="rag_collection", embedding_function=custom_embedder)


for i, chunk in enumerate(texts):
    print(chunk)


for i, chunk in enumerate(texts):
    vector = custom_embedder(texts)
    documents_list.append(chunk)
    embeddings_list.append(vector)
    ids_list.append(f"petPDF_{i}")

collection.add(
    embeddings=embeddings_list,
    documents=documents_list,
    ids=ids_list
)
"""
