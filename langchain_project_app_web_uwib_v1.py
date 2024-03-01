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


question = "what are tiers USPSTF medication?"#"what is difference in in-network and out-of-network services?" #"how to compare the prescription costs?" #"tell me more about florida blue offerd Rewards.?" #"Can you summarize the member policy]=""?" #

template = """Instructions: Use only the following context to answer the question. Give very specific answer and dont give explanation when not asked.  

Question: {question}

Context: {context}

answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
llm = CustomLLM()
db = Chroma(persist_directory="./chromadb_uwib5", embedding_function=embedder)

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       #chain_type_kwargs={'prompt': prompt}
                                        chain_type_kwargs={"verbose": True,'prompt': prompt},
                                        verbose=True
                                       )
response = qa_chain({'query': question})
print(response['result'])