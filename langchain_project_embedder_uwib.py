from langchain_custom_embedders import CustomChromaEmbedder
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

import PyPDF2

# Open the PDF file in read binary mode
pdf_file = open('./uwib_data/ValueScriptRxMedGuide_myblue.pdf', 'rb')

# Create a PdfFileReader object
pdf_reader = PyPDF2.PdfReader(pdf_file)

# Initialize an empty string to store the extracted text
output_text = ""

#page = pdf_reader.pages[1]

# Read pages 1 to 17
for page_num in range(1,15,1):
    page = pdf_reader.pages[page_num] #getPage(page_num)
    output_text += page.extract_text()
#
# # Close the PDF file
#print(output_text)

textFilepath = "./uwib_data/myBlue_rx"

loader = TextLoader(textFilepath)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
# embedding_function = CustomChromaEmbedder()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': 'cpu'})

db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chromadb_uwib1")
db.persist()
