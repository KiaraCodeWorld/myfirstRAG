from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_customllm import CustomLLM
from langchain import HuggingFacePipeline
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_customllm_hf import CustomLLM
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoder


import streamlit as st
from streamlit_chat import message
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_customllm_hf import CustomLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.llms import BaseLLM,LLMChain, HypotheticalDocumentEmbedder
from langchain_community.llms import BaseLLChain
from langchain.prompts import PromptTemplate

#member_api = requests.get("http://localhost:8001/user").text.replace("{", '<').replace("}", '>')
#{format_instructions}
template = """Answer the question based only on the  context only, if information not available in context then mention "no information available".

### Formatting ###
for all the responses, format the response in clear and bullet point summary.Make sure information is per context and no additional details be added.

Question: {question}
Context: {context}

answer:
"""

llm = CustomLLM()
def hypothetical_answer(query):
    print(query)
    messages = f"""
            You are a helpful expert Machine Learning and Deep Learning assistant.
            Provide an example answer to the given question,
            that might be found in a documents like from Book related to machine Learning and Deep Learning.

            The Question about which you have to give example answer is: {query}
        """
    question = query
    llmChain = LLMChain(llm=llm)
    response = llmChain({'query': question})
    return response

print(hypothetical_answer("what is medication guide?"))