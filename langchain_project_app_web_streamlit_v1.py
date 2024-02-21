from langchain_customllm import CustomLLM
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


question = "What are members copays as per the policy?"#"what is difference in in-network and out-of-network services?" #"how to compare the prescription costs?" #"tell me more about florida blue offerd Rewards.?" #"Can you summarize the member policy]=""?" #

#uvicorn member_api:app --host 0.0.0.0 --port 8001
member_api = requests.get("http://localhost:8001/user").text.replace("{", '<').replace("}", '>')

template = """Instructions: Use only the following context to answer the question. Give very specific answer and dont give explanation when not asked.  
Question: {question}
Context: {context} and """ + f""" member policy details are as below : {member_api}
answer:
"""
#and """ + f""" member policy details are as below : {member_api}

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
llm = CustomLLM()
db = Chroma(persist_directory="./chromadb_flblue3", embedding_function=embedder)

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 1}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
response = qa_chain({'query': question})
print(response['result'])