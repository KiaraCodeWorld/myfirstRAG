from langchain_custom_embedders import CustomChromaEmbedder
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveJsonSplitter
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
pdf_file = open('./uwib_data/ValueScriptRxMedGuide_myblue.pdf', 'rb')
#
# # Create a PdfFileReader object
pdf_reader = PyPDF2.PdfReader(pdf_file)
#
# # Initialize an empty string to store the extracted text
output_text = ""
#

# # Read pages 1 to 17
for page_num in range(1,15,1):
     page = pdf_reader.pages[page_num] #getPage(page_num)
     output_text += page.extract_text()
# #
# # # Close the PDF file
print("printing ....")
#print(output_text)
#

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks_1 = text_splitter.create_documents([output_text])
#print(len(chunks_1))

# #embedding_function = CustomChromaEmbedder()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
 model_kwargs={'device': 'cpu'})
# # # #

# def metadata_func(record: dict, metadata: dict) -> dict:
#     metadata["Drug_name"] = record.get("Drug_name")
#     return metadata
#
# Filepath = "./uwib_data/mybluelist.json"
# data = json.loads(Path(Filepath).read_text())
#
# #pprint(data)
# loader = JSONLoader(
#     file_path= Filepath,
#     jq_schema= '.records[]',
#     text_content=False,
#     metadata_func=metadata_func
# )
#
# documents = loader.load()
#chunks_2 = text_splitter.split_documents(documents)

#db = Chroma(embedding_function=embeddings, persist_directory="./chromadb_uwib4",collection_name="uwib_rx4")
db = Chroma.from_documents(documents=chunks_1, embedding=embeddings, persist_directory="./chromadb_uwib5",collection_name="uwib_rx5")
db.persist()

#db.add_documents(documents=chunks_1)
#db.add_documents(documents=chunks_2)

query = "what is USPSTF?"
print("searching in chroma")
docs = db.similarity_search(query=query)

print(docs)