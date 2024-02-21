from langchain_custom_embedders import CustomChromaEmbedder
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import PyPDF2
from langchain_community.embeddings import HuggingFaceEmbeddings


if __name__ == "__main__":

    DATA_PATH = "./data"
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = loader.load()
    text = text_splitter.split_documents(documents)
    embedding_function = CustomChromaEmbedder()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = Chroma.from_documents(documents=text, embedding=embeddings , persist_directory="./chromadb_flblue2")
    db.persist()

    """web_data = pdf_to_text("./data/pet.pdf")
    chunks = text_splitter.split_text(web_data)
    customer_embedder = CustomChromaEmbedder() """
    # printConvert chunks to vector representations and store in Chroma DB

    #db = Chroma.from_documents(documents=text, embedding=embedding_function, persist_directory="./chromadb_flblue2")
    #print("embedding complete")