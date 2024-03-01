from langchain_custom_embedders import CustomChromaEmbedder
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

import PyPDF2
from langchain_custom_embedders import CustomChromaEmbedder
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, JSONLoader
import PyPDF2
import json
from pathlib import Path
from pprint import pprint
from langchain.text_splitter import RecursiveJsonSplitter

# # Open the PDF file in read binary mode
# pdf_file = open('./uwib_data/ValueScriptRxMedGuide_myblue.pdf', 'rb')
#
# # Create a PdfFileReader object
# pdf_reader = PyPDF2.PdfReader(pdf_file)
#
# # Initialize an empty string to store the extracted text
# output_text = ""
#
# #page = pdf_reader.pages[1]
#
# # Read pages 1 to 17
# for page_num in range(1,15,1):
#     page = pdf_reader.pages[page_num] #getPage(page_num)
#     output_text += page.extract_text()
# #
# # # Close the PDF file
# #print(output_text)

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["Drug_name"] = record.get("Drug_name")
    return metadata

Filepath = "./uwib_data/mybluelist.json"
data = json.loads(Path(Filepath).read_text())

#pprint(data)

loader = JSONLoader(
    file_path= Filepath,
    jq_schema= '.records[]',
    text_content=False,
    metadata_func=metadata_func
)

documents = loader.load()

#data = json.loads(Path(textFilepath).read_text())
# pprint(Path(textFilepath).read_text())
#
# loader = TextLoader(
#     file_path=textFilepath
# )
#     # jq_schema='.content',
#     # text_content=False,
#     # json_lines=True)
#     # json_lines=True)
#     # json_lines=True)
#
# documents = loader.load()
# #
# # loader = JSONLoader(textFilepath)
# # documents = loader.load()
# #

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100 ) #, separators="}")
chunks = text_splitter.split_documents(documents)

#text_splitter = RecursiveJsonSplitter(max_chunk_size=300)
#chunks = text_splitter(documents)

#print(chunks)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                     model_kwargs={'device': 'cpu'})

model =  "BAAI/bge-large-en" #"BAAI/bge-small-en"
#embeddings = HuggingFaceEmbeddings(model_name=model,
#                                     model_kwargs={'device': 'cpu'})

print("writing to chroma")
db = Chroma.from_documents(documents=chunks, collection_metadata={"hnsw:space": "cosine"}, embedding=embeddings, persist_directory="./chromadb_uwib3",collection_name="uwib_rx3")
db.persist()
#db = Chroma(persist_directory="./chromadb_uwib3", embedding_function=embeddings)

print("writing to chroma is done...")

query = "give me information about OFEV"
docs = db.similarity_search(query=query)
print("searching in chroma")

print(docs)