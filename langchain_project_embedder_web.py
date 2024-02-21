from langchain_custom_embedders import CustomChromaEmbedder
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

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
                        "https://www.floridablue.com/answers/money-saving-tips/saving-on-imaging-services",
                        "https://www.floridablue.com/answers/money-saving-tips/knowing-where-to-go-for-care",

                        # how to pickup plan
                        "https://www.floridablue.com/answers/how-to-pick-a-plan/prescription-coverage"
                        
                        # how coverage works 
                        "https://www.floridablue.com/answers/health-coverage-basics/how-health-coverage-works",
                        "https://www.floridablue.com/answers/health-coverage-basics/in-network-versus-out-of-network",
                        "https://www.floridablue.com/answers/health-coverage-basics/where-can-i-find-my-yearly-health-coverage-tax-form",
                        "https://www.floridablue.com/answers/health-coverage-basics/buying-a-plan-from-the-health-insurance-marketplace",
                        "https://www.floridablue.com/answers/health-coverage-basics/the-essentials",
                        "https://www.floridablue.com/answers/health-coverage-basics/preexisting-conditions",
                        "https://www.floridablue.com/answers/health-coverage-basics/medical-loss-ratio"]

    web_loader = WebBaseLoader(web_url_list, default_parser="html.parser" )
    web_loader.requests_per_second = 2
    web_data = web_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(web_data)
    #embedding_function = CustomChromaEmbedder()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = Chroma.from_documents(documents=chunks, embedding=embeddings , persist_directory="./chromadb_flblue3")
    db.persist()
