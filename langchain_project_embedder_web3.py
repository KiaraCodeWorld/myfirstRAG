from langchain_custom_embedders import CustomChromaEmbedder
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_custom_embedders_v3 import default_embedder

if __name__ == "__main__":
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
    """
    web_url_list = [ # money saving tips
                        "https://www.floridablue.com/answers/money-saving-tips/comparing-medical-and-prescription-costs",
                        "https://www.floridablue.com/answers/money-saving-tips/discounts-and-rewards",
                        "https://www.floridablue.com/answers/money-saving-tips/saving-on-imaging-services"
                       ]

    web_loader = WebBaseLoader(web_url_list, default_parser="html.parser" )
    web_loader.requests_per_second = 2
    web_data = web_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=50)
    texts = text_splitter.split_documents(web_data)
    #embedding_function = CustomChromaEmbedder()
    embeddings = default_embedder

    db = Chroma.from_documents(documents=texts, embedding=embeddings , persist_directory="./chromadb_flblue6")
    db.persist()