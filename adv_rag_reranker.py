from langchain.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
#from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from sentence_transformers import CrossEncoder

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

documents = TextLoader("./state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
embeddingsModel = HuggingFaceEmbeddings(
    model_name="sentence-transformers/msmarco-distilbert-dot-v5"
)
retriever = FAISS.from_documents(texts, embeddingsModel).as_retriever(
    search_kwargs={"k": 20}
)

query = "What is the plan for the economy?"
docs = retriever.invoke(query)
#print(docs)
#pretty_print_docs(docs)

from langchain.retrievers import ContextualCompressionRetriever
from sentence_transformers.cross_encoder import CrossEncoder
#from langchain.retrievers.document_compressors import CrossEncoderReranker
#from langchain_community.document_loaders import CrossEncoderReranker, HuggingFaceCrossEncoder

#model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker_model = CrossEncoder(model_name="BAAI/bge-reranker-large", max_length=512)

def rerank_docs(query, retrieved_docs):
    query_and_docs = [(query, r.page_content ) for i,r in enumerate(retrieved_docs)]
    scores = reranker_model.predict(query_and_docs)
    return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)

#print([f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)])
docs2 = rerank_docs(query, docs)
print(docs2)

p#=[]'
''# docs2 = rerank_docs(query, docs)
#pretty_print_docs(docs2)

#compressor = CrossEncoderReranker(model=model, top_n=3)
#compression_retriever = ContextualCompressionRetriever(
#    base_compressor=compressor, base_retriever=retriever

#compressed_docs = compression_retriever.invoke("What is the plan for the economy?")
#pretty_print_docs(compressed_docs)