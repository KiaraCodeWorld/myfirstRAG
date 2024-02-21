from langchain_custom_embedders import CustomChromaEmbedder
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import tabula as tb
from langchain.document_loaders.parsers.pdf import PDFPlumberParser
from dotenv import load_dotenv
load_dotenv()
import os
openai_api_key = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":

    from typing import Any

    from pydantic import BaseModel
    from unstructured.partition.pdf import partition_pdf

    path = "./uwib_data/ValueScriptRxMedGuide_split.pdf"  #PDFwithTable.pdf" #ValueScriptRxMedGuide_myblue.pdf

    # Get elements
    raw_pdf_elements = partition_pdf(
        filename=path,
        # Unstructured first finds embedded image blocks
        extract_images_in_pdf=False,
        # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
        # Titles are any sub-section of the document
        infer_table_structure=True,
        # Post processing to aggregate text once we have the title
        chunking_strategy="by_title",
        # Chunking params to aggregate text blocks
        # Attempt to create a new chunk 3800 chars
        # Attempt to keep chunks > 2000 chars
        max_characters=4000,
        new_after_n_chars=3800,
        model_name = "yolox",
        combine_text_under_n_chars=2000,
        image_output_dir_path=path,
    )

    # Create a dictionary to store counts of each type
    category_counts = {}

    for element in raw_pdf_elements:
        category = str(type(element))
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1

    # Unique_categories will have unique elements
    unique_categories = set(category_counts.keys())
    print(category_counts)


    class Element(BaseModel):
        type: str
        text: Any

    # Categorize by type
    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))

    # Tables
    table_elements = [e for e in categorized_elements if e.type == "table"]
    print(len(table_elements))
    #print(table_elements)

    # Text
    text_elements = [e for e in categorized_elements if e.type == "text"]
    print(len(text_elements))

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text. \ 
    Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatOpenAI(temperature=0, model="gpt-4")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()


    print("**** Summarize tables")
    # Apply to tables
    tables = [i.text for i in table_elements]
    #print(tables)
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    print(table_summaries)


    print("**** multivector processing *******")
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text. \ 
    Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatOpenAI(temperature=0, model="gpt-4")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Apply to texts
    texts = [i.text for i in text_elements]
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})

    import uuid

    from langchain.retrievers.multi_vector import MultiVectorRetriever
    from langchain.storage import InMemoryStore
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())

    # The storage layer for the parent documents
    store = InMemoryStore()
    id_key = "doc_id"

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=s, metadata={id_key: table_ids[i]})
        for i, s in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))

    from langchain_core.runnables import RunnablePassthrough

    # Prompt template
    template = """Answer the question based only on the following context, which can include text and tables:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4")

    # RAG pipeline
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )

    response = chain.invoke("Give me details about how Drug Name, its tier, other details for BAXDELA?")
    print(response)


