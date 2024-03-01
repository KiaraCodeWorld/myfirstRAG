#from langchain_customllm import CustomLLM
from langchain_customllm_hf import CustomLLM
from langchain_custom_embedders import CustomChromaEmbedder
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import requests
import streamlit as st, streamlit_chat as sc
from streamlit_chat import message

# #embedding_function = CustomChromaEmbedder()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
 model_kwargs={'device': 'cpu'})
# # # #

db = Chroma(persist_directory="./chromadb_uwib7", embedding_function=embeddings)

query = "Drug_name: cyproheptadine hcl syrup" #"why medication is not covered?" #"Drug_name: cyproheptadine hcl syrup" #"Drug_name: cyproheptadine hcl syrup"
#VYNDAMAX" #"what are medication tiers?" #"what is ValueScript Rx Medication Guide"
print("searching in chroma")
docs = db.similarity_search(query=query)

print(docs)


#KEVZARA - sarilumab subcutaneous