from langchain_customLLM_v1 import CustomLLM

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage,SystemMessage, AIMessage
from langchain_custom_embeddings_chroma_v1 import CustomEmbeddings

# load the document
loader = TextLoader("./sou1.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)

print(chunks)

#embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

custom_embeddings = CustomEmbeddings
db = Chroma.from_documents(documents=chunks, embedding=custom_embeddings, persist_directory="./chroma_db")
