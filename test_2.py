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
from langchain_customllm_hf import CustomLLM
from langchain.chains import RetrievalQA
import requests
import streamlit as st, streamlit_chat as sc
from streamlit_chat import message
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

metadata_field_info = [
    AttributeInfo(
        name="Drug_name",
        description="Name of the medication or drug",
        type="string",
    )
]

# #embedding_function = CustomChromaEmbedder()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
 model_kwargs={'device': 'cpu'})

llm = CustomLLM()
db = Chroma(persist_directory="./chromadb_uwib6", embedding_function=embeddings)

# Instantiate the self-querying retriever
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=db,
    document_contents="details of the medication, tier, requirement/limits etc",
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    search_kwargs={"k": 10},
    verbose=True
     # Optional: Limit the number of retrieved documents
)

print("printing results.....")
query = "information about medication name" #Drug_name = KAMELEON "
results = retriever.invoke(query)
print(results)  # Display the retrieved documents
