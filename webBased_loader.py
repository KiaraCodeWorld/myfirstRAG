import requests
from bs4 import BeautifulSoup
from langchain_customLLM_v1 import CustomLLM
from langchain.chains import LLMChain
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage,SystemMessage, AIMessage
from langchain_custom_embeddings_v1 import CustomEmbeddings

# Define the URL of the parent page
web_url_list = ["https://www.floridablue.com/answers/money-saving-tips/comparing-medical-and-prescription-costs",
            "https://www.floridablue.com/answers/money-saving-tips/discounts-and-rewards",
            "https://www.floridablue.com/answers/money-saving-tips/saving-on-imaging-services"]

web_loader = WebBaseLoader(web_url_list, default_parser="html.parser")
web_loader.requests_per_second = 1
web_data = web_loader.load()
print(web_data[0].page_content)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
chunks =  text_splitter.split_documents(web_data)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, embedding_function ) #, persist_directory="./flblue_chroma")

question = "Who won the FIFA World Cup in the year 1994? "
context =  db.similarity_search(question) #"Portugal won the World Cup in 1994."

print(context)

template = """Instructions: Use only the following context to answer the question.
Context: {context}
Question: {question}

"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

llm = CustomLLM(n=50)
llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.invoke({"context": context, "question": question}))

#print(llm_chain.invoke({"context": context, "question": question}))
