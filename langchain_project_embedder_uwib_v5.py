from langchain_custom_embedders import CustomChromaEmbedder
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, CSVLoader

import PyPDF2
from langchain_custom_embedders import CustomChromaEmbedder
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, JSONLoader
from CsvMetadataLoader import CsvMetadataLoader

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

"""
Loading of the myBlue.csv which has all the Rx, their tiers, requirements/limits, Speciality etc.
"""
csvfilepath = "./uwib_data/myBlue.csv"

loader_rx = CsvMetadataLoader(file_path=csvfilepath, metadata_columns=['Drug_name'])
documents_rx = loader_rx.load()

chunks_rx = text_splitter.split_documents(documents_rx)


"""
Loading of the PDF, from page 2 to 17 only containing the information
"""
# Open the PDF file in read binary mode
myBlueRx_pdf = open('./uwib_data/ValueScriptRxMedGuide_myblue.pdf', 'rb')

# Create a PdfFileReader object
pdf_reader = PyPDF2.PdfReader(myBlueRx_pdf)

# Initialize an empty string to store the extracted text
output_text = ""

# Read pages 1 to 17
for page_num in range(1,15,1):
    page = pdf_reader.pages[page_num] #getPage(page_num)
    output_text += page.extract_text()

chunks_pdf = text_splitter.create_documents([output_text])

"""
Initialise the chroma db and add documents 
"""
#define the embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': 'cpu'})

#db = Chroma(embedding_function=embeddings, persist_directory="./chromadb_uwib7",collection_name="uwib_myblue_rx")
db = Chroma.from_documents(documents=chunks_rx, embedding=embeddings, persist_directory="./chromadb_uwib7") #,collection_name="uwib_myblue_rx")

db.add_documents(documents=chunks_pdf)
db.persist()

#db.add_documents(documents=chunks_rx)
#db.add_documents(documents=chunks_pdf)

# query = "what are medication tiers?"
# print("searching in chroma")
# docs = db.similarity_search(query=query)
#
# print(docs)





import PyPDF2

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
#
# textFilepath = "./uwib_data/myBlue_rx"
#
# loader = TextLoader(textFilepath)
# documents = loader.load()

