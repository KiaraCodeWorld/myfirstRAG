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

csvfilepath = "./uwib_data/myBlue.csv"

loader = CsvMetadataLoader(file_path=csvfilepath, metadata_columns=['Drug_name'])
#loader = CSVLoader(file_path=csvfilepath, source_column="Drug_name" , metadata_columns=['Drug_name'])
#loader = CSVLoader(file_path=csvfilepath, source_column=['Drug Tier','Specialty','Plan Type','Allowed','Requirements_Limits'], metadata_columns=['Drug_name'])
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

#embedding_function = CustomChromaEmbedder()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    model_kwargs={'device': 'cpu'})
#
db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chromadb_uwib6")
db.persist()