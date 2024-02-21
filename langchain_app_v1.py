from langchain_customLLM_v1 import CustomLLM

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage,SystemMessage, AIMessage
from langchain_custom_embeddings_v1 import CustomEmbeddings

# load the document
loader = TextLoader("./sou1.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)

print(chunks)


# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(chunks, embedding_function)

# query it
#query = "how many new jobs were created?"
#docs = db.similarity_search(query)

# print results
#print(docs[0].page_content)

print("****************")

question = "Who won the FIFA World Cup in the year 1994? "
docs = db.similarity_search(question)
context =  "Portugal won the World Cup in 1994."

template = """Instructions: Use only the following context to answer the question.
Context: {context}
Question: {question}

"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

llm = CustomLLM(n=50)
llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.invoke({"context": context, "question": question}))