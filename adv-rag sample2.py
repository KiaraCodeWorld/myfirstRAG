from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.vectorstores import Redis
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from typing import Optional, List, Mapping, Any
import gradio as gr
import requests
import numpy as np
from sentence_transformers import CrossEncoder

# Configuration
REDIS_URL = "redis://localhost:6379"
INDEX_NAME = "arag_coll"
LLM_API = "https://api.customllm.com/v1/llmam3"
EMBED_API = "https://api.customembed.com/v1/embeddings"
API_KEY = "your-api-key-here"

class CustomLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, 
             temperature: float = 0.7, top_p: float = 0.9) -> str:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.post(
            LLM_API,
            headers=headers,
            json={
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": 500
            }
        )
        return response.json()["generations"][0]["text"]

class CustomEmbedder(Embeddings):
    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            EMBED_API,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"inputs": [text]}
        )
        return response.json()["embeddings"][0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            EMBED_API,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"inputs": texts}
        )
        return response.json()["embeddings"]

# Initialize components
llm = CustomLLM()
embeddings = CustomEmbedder()

# Connect to Redis vector store
vectorstore = Redis.from_existing_index(
    embedding=embeddings,
    index_name=INDEX_NAME,
    redis_url=REDIS_URL
)

# Advanced RAG Components
hyde_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Generate a detailed hypothetical answer to this question:
Question: {question}
Hypothetical Answer:"""
)

query_expansion_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Generate 5 different reformulations of this question covering various aspects:
Original Question: {question}
Reformulated Questions:"""
)

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

def hyde_retriever(query: str, k: int = 5) -> List[Document]:
    hyde_chain = LLMChain(llm=llm, prompt=hyde_prompt)
    hypothetical_answer = hyde_chain.run(query)
    return vectorstore.similarity_search(hypothetical_answer, k=k)

def expand_queries(query: str) -> List[str]:
    expansion_chain = LLMChain(llm=llm, prompt=query_expansion_prompt)
    expanded = expansion_chain.run(query).strip().split('\n')
    return [q.split(' ', 1)[1] for q in expanded[:5] if q.strip()]

def hybrid_search(query: str, k: int = 10) -> List[Document]:
    # Vector search
    vector_results = vectorstore.similarity_search(query, k=k)
    
    # Hybrid search with expanded queries
    expanded_queries = expand_queries(query)
    all_queries = [query] + expanded_queries
    
    results = []
    for q in all_queries:
        results.extend(vectorstore.similarity_search(q, k=2))
        results.extend(hyde_retriever(q, k=2))
    
    # Deduplicate documents
    seen = set()
    unique_docs = []
    for doc in results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    return unique_docs

def rerank_documents(query: str, docs: List[Document]) -> List[Document]:
    document_contents = [doc.page_content for doc in docs]
    scores = cross_encoder.predict([(query, doc) for doc in document_contents])
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs]

def format_response(response: str) -> str:
    return "\n".join([f"• {line}" for line in response.split('\n') if line.strip()])

def rag_pipeline(query: str) -> str:
    # Retrieve documents
    documents = hybrid_search(query)
    
    # Rerank documents
    ranked_docs = rerank_documents(query, documents)[:5]
    
    # Generate answer
    context = "\n\n".join([d.page_content for d in ranked_docs])
    response = llm(f"""Answer this question using only the context below. 
                  Use bullet points for key information and keep it concise.
                  
                  Question: {query}
                  Context: {context}
                  
                  Answer:""")
    
    return format_response(response)

# Gradio Chat Interface
def chat(message: str, history: List[List[str]]):
    response = rag_pipeline(message)
    return response

demo = gr.ChatInterface(
    fn=chat,
    title="Advanced RAG Chatbot",
    description="Ask questions and get answers with enhanced retrieval capabilities",
    examples=[
        "Explain the key features of our product?",
        "What are the main advantages of your solution?",
        "How does your system handle security concerns?"
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()

#### ========
https://www.kaggle.com/code/usman49/advance-rag-using-llama2-langchain-chromadb


from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Redis
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from typing import Optional, List, Mapping, Any
import gradio as gr
import requests
import numpy as np
from sentence_transformers import CrossEncoder
import re

# Configuration
REDIS_URL = "redis://localhost:6379"
INDEX_NAME = "arag_coll"
LLM_API = "https://api.customllm.com/v1/llmam3"
EMBED_API = "https://api.customembed.com/v1/embeddings"
VALIDATION_API = "https://api.validation.com/v1/check"
API_KEY = "your-api-key-here"

class CustomLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, 
             temperature: float = 0.7) -> str:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.post(
            LLM_API,
            headers=headers,
            json={"prompt": prompt, "temperature": temperature}
        )
        return response.json()["generations"][0]["text"]

class CustomEmbedder(Embeddings):
    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            EMBED_API,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"text": text}
        )
        return response.json()["embedding"]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

# Initialize components
llm = CustomLLM()
embeddings = CustomEmbedder()
vectorstore = Redis.from_existing_index(embeddings, INDEX_NAME, REDIS_URL)

# Advanced RAG Components
hyde_prompt = PromptTemplate(
    input_variables=["question"],
    template="Generate a comprehensive hypothetical answer to: {question}"
)

expansion_prompt = PromptTemplate(
    input_variables=["question"],
    template="Generate 5 diverse reformulations of: {question}"
)

validation_prompt = PromptTemplate(
    input_variables=["answer", "context"],
    template="""Verify if this answer is fully supported by the context. 
    Identify any hallucinations or unsupported claims:
    Answer: {answer}
    Context: {context}
    Validation Report:"""
)

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

def hyde_retrieval(query: str) -> List[Document]:
    hyde_chain = LLMChain(llm=llm, prompt=hyde_prompt)
    hypothetical_answer = hyde_chain.run(query)
    return vectorstore.similarity_search(hypothetical_answer, k=3)

def expand_queries(query: str) -> List[str]:
    expansion_chain = LLMChain(llm=llm, prompt=expansion_prompt)
    expanded = expansion_chain.run(query).strip().split('\n')
    return [q.split('. ', 1)[1] for q in expanded[:5] if q.strip()]

def hybrid_search(query: str) -> List[Document]:
    base_docs = vectorstore.similarity_search(query, k=3)
    hyde_docs = hyde_retrieval(query)
    expanded_queries = expand_queries(query)
    expanded_docs = [vectorstore.similarity_search(q, k=2) for q in expanded_queries]
    
    all_docs = base_docs + hyde_docs + [doc for sublist in expanded_docs for doc in sublist]
    seen = set()
    return [doc for doc in all_docs if not (doc.page_content in seen or seen.add(doc.page_content))]

def rerank_documents(query: str, docs: List[Document]) -> List[Document]:
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:5]]

def validate_response(answer: str, context: str) -> dict:
    validation_chain = LLMChain(llm=llm, prompt=validation_prompt)
    report = validation_chain.run({"answer": answer, "context": context})
    
    # Calculate hallucination score
    hallucination_keywords = ["unsupported", "no evidence", "cannot confirm", "speculative"]
    score = sum(1 for word in hallucination_keywords if word in report.lower()))
    
    return {
        "report": report,
        "hallucination_score": score,
        "is_valid": score < 2
    }

def format_response(response: str) -> str:
    return "\n".join([f"• {line.strip()}" for line in response.split('\n') if line.strip()])

def rag_pipeline(query: str) -> tuple:
    # Retrieve and process documents
    documents = hybrid_search(query)
    ranked_docs = rerank_documents(query, documents)
    context = "\n\n".join([doc.page_content for doc in ranked_docs])
    
    # Generate answer
    answer = llm(f"Answer this query concisely using the context below:\n\nQuery: {query}\nContext: {context}")
    formatted_answer = format_response(answer)
    
    # Validate response
    validation = validate_response(answer, context)
    
    return formatted_answer, validation["report"], validation["hallucination_score"]

# Gradio Interface
with gr.Blocks(title="Advanced RAG Chatbot") as demo:
    gr.Markdown("# Advanced RAG Chatbot with Validation")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation")
            msg = gr.Textbox(label="Your Question")
            examples = gr.Examples(
                examples=["Explain key features", "Technical specifications", "Implementation guidelines"],
                inputs=msg
            )
        
        with gr.Column(scale=2):
            validation_output = gr.Textbox(label="Validation Report", lines=8)
            score_display = gr.Number(label="Hallucination Score")
    
    def respond(query, chat_history):
        answer, report, score = rag_pipeline(query)
        chat_history.append((query, answer))
        return chat_history, report, score
    
    msg.submit(
        respond,
        [msg, chatbot],
        [chatbot, validation_output, score_display],
        queue=False
    )

if __name__ == "__main__":
    demo.launch()

###########


import os
import requests
from langchain.vectorstores.redis import Redis
from langchain.retrievers import ContextualCompressionRetriever, SelfQueryRetriever
from langchain.retrievers.hybrid import HybridRetriever
from langchain.retrievers.hyde import HydeRetriever
from langchain.schema import BaseRetriever, Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks import StdOutCallbackHandler
import gradio as gr

# Configuration
REDIS_URL = "redis://localhost:6379"
COLLECTION_NAME = "arag_coll"
LLM_API_URL = os.getenv("LLM_AM3_API_URL")
EMBEDDING_API_URL = os.getenv("CUSTOM_EMBEDDING_API_URL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Custom LLM implementation [[3]][[6]]
class CustomLLM(LLM):
    def _call(self, prompt: str, stop=None) -> str:
        response = requests.post(
            LLM_API_URL,
            json={"prompt": prompt, "max_tokens": 500},
            headers={"Authorization": f"Bearer {os.getenv('LLM_API_KEY')}"}
        )
        return response.json()['generated_text']
    
    @property
    def _llm_type(self) -> str:
        return "custom_llm"

# Custom Embedding model [[1]][[4]]
class CustomEmbedder(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            EMBEDDING_API_URL,
            json={"texts": texts},
            headers={"Authorization": f"Bearer {os.getenv('EMBEDDING_API_KEY')}"}
        )
        return response.json()['embeddings']

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Initialize components [[1]][[4]]
llm = CustomLLM()
embedder = CustomEmbedder()

# Redis vector store setup [[4]][[7]]
redis_store = Redis.from_existing_index(
    embedding=embedder,
    redis_url=REDIS_URL,
    index_name=COLLECTION_NAME
)

# Hybrid search setup [[9]]
bm25_retriever = BM25Retriever.from_documents(redis_store.get_all_documents())
vector_retriever = redis_store.as_retriever()
hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

# Query expansion with SelfQueryRetriever [[2]]
metadata_fields = [
    AttributeInfo(name="content", description="Document content", type="string")
]
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=redis_store,
    document_contents="Document content",
    metadata_field_info=metadata_fields
)

# Re-ranking with Cohere [[9]]
compression_retriever = ContextualCompressionRetriever(
    base_compressor=CohereReranker(model='rerank-english-v2.0', api_key=COHERE_API_KEY),
    base_retriever=hybrid_retriever
)

# HYDE implementation [[7]]
hyde_retriever = HydeRetriever(
    llm=llm,
    base_retriever=compression_retriever
)

# Validation prompt [[8]]
validation_prompt = PromptTemplate(
    template="""Verify if the answer is supported by the context:
    Context: {context}
    Answer: {answer}
    Response:""",
    input_variables=["context", "answer"]
)

# RAG chain setup [[6]]
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=hyde_retriever,
    return_source_documents=True
)

# Hallucination detection [[8]]
def detect_hallucinations(context: List[Document], answer: str) -> str:
    validation_chain = validation_prompt | llm
    result = validation_chain.invoke({
        "context": "\n".join([doc.page_content for doc in context]),
        "answer": answer
    })
    return "Hallucination detected" if "no" in result.lower() else "Valid response"

# Gradio interface [[5]]
def rag_interface(query: str) -> str:
    result = qa_chain.invoke(query)
    validation = detect_hallucinations(
        context=result['source_documents'],
        answer=result['result']
    )
    return f"Answer: {result['result']}\n\nValidation: {validation}"

iface = gr.Interface(
    fn=rag_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query..."),
    outputs="text",
    title="Advanced RAG System"
)

if __name__ == "__main__":
    iface.launch()

