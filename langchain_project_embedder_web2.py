from langchain_custom_embedders import CustomChromaEmbedder
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import PyPDF2
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import UnstructuredURLLoader
import xml.etree.ElementTree as ET
from loguru import logger
import requests

def extract_urls_from_sitemap(sitemap):
    """
    Extract all URLs from a sitemap XML string.

    Args:
        sitemap_string (str): The sitemap XML string.

    Returns:
        A list of URLs extracted from the sitemap.
    """
    # Parse the XML from the string
    root = ET.fromstring(sitemap)

    # Define the namespace for the sitemap XML
    namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # Find all <loc> elements under the <url> elements
    urls = [
        url.find("ns:loc", namespace).text for url in root.findall("ns:url", namespace)
    ]

    # Return the list of URLs
    return urls


if __name__ == "__main__":
    """
    print("hello")
    sitemap_url = "https://www.floridablue.com/sitemap.xml"
    sitemap = requests.get(sitemap_url).text
    urls = extract_urls_from_sitemap(sitemap)
    print(urls)
    """
    sitemap_url = "https://www.floridablue.com/sitemap.xml"

    print("Building the knowledge base ...")

    print(f"Loading sitemap from {sitemap_url} ...")
    sitemap = requests.get(sitemap_url).text
    urls = extract_urls_from_sitemap(sitemap)
    pattern = "education"

    #print(urls)

    if pattern:
        print(f"Filtering URLs with pattern {pattern} ...")
        urls = [x for x in urls if pattern in x]
    #print("{n} URLs extracted", n=len(urls))

    webs = [  # money saving tips
        "https://www.floridablue.com/answers/money-saving-tips/comparing-medical-and-prescription-costs",
        "https://www.floridablue.com/answers/money-saving-tips/discounts-and-rewards" ]

    print("Loading URLs content ...")
    #loader = UnstructuredURLLoader(webs)
    #data = loader.load()
    #print(data)

    print(urls)

    web_loader = WebBaseLoader(urls, default_parser="html.parser")
    web_loader.requests_per_second = 2
    web_data = web_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(web_data)
    # embedding_function = CustomChromaEmbedder()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chromadb_flblue_edu")
    db.persist()
