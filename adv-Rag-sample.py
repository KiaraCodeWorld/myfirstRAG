from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.vectorstores import Redis
from langchain.schema import Document
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from typing import Optional, List, Mapping, Any
import gradio as gr
import requests
import numpy as np
from sentence_transformers import CrossEncoder

# Configuration
REDIS_URL = "redis://localhost:6379"
INDEX_NAME = "arag_coll"
API_ENDPOINT = "https://your-llm-api-endpoint.com/v1/llmam3"
EMBED_API = "https://your-embed-api-endpoint.com/embed"
HEADERS = {"Authorization": "Bearer your_api_key"}

# Custom LLM Implementation
class CustomLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, temperature: float = 0.7) -> str:
        response = requests.post(
            API_ENDPOINT,
            headers=HEADERS,
            json={"prompt": prompt, "temperature": temperature}
        )
        return response.json()["response"]

# Custom Embeddings Implementation
class CustomEmbedder(Embeddings):
    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            EMBED_API,
            headers=HEADERS,
            json={"text": text}
        )
        return response.json()["embedding"]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings

# Initialize components
llm = CustomLLM()
embeddings = CustomEmbedder()

# Connect to Redis vector store
vectorstore = Redis.from_existing_index(
    embedding=embeddings,
    index_name=INDEX_NAME,
    redis_url=REDIS_URL
)

# HYDE Document Generator
hyde_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Please write a detailed hypothetical answer to the following question. 
    The answer should be comprehensive and self-contained.
    Question: {question}
    Hypothetical Answer:"""
)
hyde_chain = LLMChain(llm=llm, prompt=hyde_prompt)

# Query Expansion Template
expansion_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Generate 5 different reformulations of the following question. 
    The new questions should be more specific and cover different aspects:
    Original Question: {question}
    Reformulated Questions:"""
)

# Cross-Encoder Reranker
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
    doc_texts = [doc.page_content for doc in documents]
    scores = cross_encoder.predict([(query, doc) for doc in doc_texts])
    sorted_indices = np.argsort(scores)[::-1]
    return [documents[i] for i in sorted_indices[:top_k]]

# Full Processing Pipeline
def process_query(query: str):
    # HYDE Generation
    hypothetical_answer = hyde_chain.run(query)
    
    # Query Expansion
    expanded_queries = llm(expansion_prompt.format(question=query)).strip().split("\n")
    all_queries = [query, hypothetical_answer] + expanded_queries[:5]
    
    # Retrieve documents
    documents = []
    for q in all_queries:
        docs = vectorstore.similarity_search(q, k=3)
        documents.extend(docs)
    
    # Rerank documents
    reranked_docs = rerank_documents(query, documents)
    
    # Generate final answer
    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    response = llm(f"""Answer the question using the provided context. 
                  Format your response as bullet points.
                  
                  Question: {query}
                  Context: {context}
                  
                  Answer in bullet points:""")
    
    return response

# Gradio Interface
def chat_interface(query: str):
    response = process_query(query)
    return response

demo = gr.Interface(
    fn=chat_interface,
    inputs=gr.Textbox(label="Enter your question"),
    outputs=gr.Markdown(label="Response"),
    title="RAG Chatbot",
    description="Ask questions and get answers powered by advanced retrieval and ranking techniques"
)

if __name__ == "__main__":
    demo.launch()
########################### sample code from perplexity - openai : 


from langchain_community.vectorstores import Redis
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.embeddings import Embeddings
from typing import List
import requests
import gradio as gr
import os

# Custom Embedding Model via API
class CustomEmbedder(Embeddings):
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": texts}
        )
        return response.json().get('embeddings', [])
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Custom LLM via API (Llama 3.3)
class CustomLLM:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def generate(self, prompt: str, **kwargs) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.3
            }
        }
        response = requests.post(self.api_url, json=payload, headers=self.headers)
        return response.json().get('generated_text', '')

# Initialize components
custom_llm = CustomLLM(
    api_url="https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct",
    api_key=os.getenv("HF_API_KEY")
)

vector_store = Redis(
    redis_url="redis://localhost:6379",
    index_name="arag_coll",
    embedding=CustomEmbedder(
        api_url="https://your-embedding-api.com/v1/embed",
        api_key=os.getenv("EMBED_API_KEY")
    )
)

# Cross-encoder setup remains same
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

# HYDE Implementation with custom LLM
hyde_prompt = ChatPromptTemplate.from_template(
    """Answer the user's question to help with document retrieval:
    
    Question: {question}
    Hypothetical Answer:"""
)
hyde_chain = hyde_prompt | custom_llm.generate | StrOutputParser()

# Query Expansion with structured output
def expand_queries(question: str) -> List[str]:
    expansion_prompt = f"""Generate 5 technical query expansions for: {question}
    Focus on code-related terms and synonyms."""
    response = custom_llm.generate(expansion_prompt)
    return [q.strip() for q in response.split("\n")[:5]]

# Processing pipeline
def process_query(question: str):
    hypothetical_answer = hyde_chain.invoke({"question": question})
    expanded_queries = [question] + expand_queries(question)
    
    docs = []
    for query in expanded_queries:
        docs.extend(vector_store.similarity_search(query, k=3))
        
    compressed_docs = reranker.compress_documents(docs, question)
    return "\n- " + "\n- ".join([d.page_content for d in compressed_docs])

# Gradio interface remains same
with gr.Blocks() as demo:
    with gr.Row():
        inp = gr.Textbox(label="Enter your question")
        out = gr.Markdown()
    btn = gr.Button("Ask")
    btn.click(fn=process_query, inputs=inp, outputs=out)

demo.launch()
############  Openai 

# Install necessary packages (if not already installed)
# !pip install langchain==0.0.187 gradio redis sentence-transformers torch

import os
import gradio as gr
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from langchain.document_loaders import RedisLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores.redis import Redis
from langchain.embeddings.base import Embeddings
from langchain.retrievers import RedisRetriever
from langchain.schema import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
import redis
import torch
import requests  # For making API calls

# Initialize Redis connection
redis_url = "redis://localhost:6379"  # Update with your Redis URL if different
redis_client = redis.Redis.from_url(redis_url)

# Constants
VECTOR_STORE_INDEX_NAME = "arag_coll"  # The Redis collection name

# Custom Embeddings class using customEmbedder API
class CustomEmbeddings(Embeddings):
    def __init__(self, api_url, api_key=None):
        self.api_url = api_url  # API endpoint for customEmbedder
        self.api_key = api_key  # If authentication is required

    def embed_query(self, text):
        # Code to call customEmbedder API and get the embedding for the text
        response = requests.post(
            self.api_url,
            json={"text": text},
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )
        embedding = response.json().get("embedding")
        return embedding

    def embed_documents(self, texts):
        # Code to call customEmbedder API in bulk and get embeddings for texts
        response = requests.post(
            self.api_url,
            json={"texts": texts},
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )
        embeddings = response.json().get("embeddings")
        return embeddings

# Custom LLM class using llmam3 API
class CustomLLM(LLM):
    def __init__(self, api_url, api_key=None, temperature=0.7):
        self.api_url = api_url  # API endpoint for llmam3
        self.api_key = api_key  # If authentication is required
        self.temperature = temperature

    @property
    def _llm_type(self):
        return "custom_llm"

    def _call(self, prompt, stop=None):
        # Code to call llmam3 API and get the response for the prompt
        payload = {
            "prompt": prompt,
            "temperature": self.temperature,
            "stop": stop if stop else [],
        }
        response = requests.post(
            self.api_url,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )
        text = response.json().get("generated_text")
        return text

# Set device for torch (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the custom embedding model
embedding_api_url = "https://api.customembedder.com/embeddings"  # Replace with your API endpoint
embedding_api_key = "YOUR_CUSTOM_EMBEDDER_API_KEY"  # Replace with your API key if needed
embedding_model = CustomEmbeddings(api_url=embedding_api_url, api_key=embedding_api_key)

# Initialize cross-encoder model for reranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Initialize HYDE LLM using the custom LLM
llm_api_url = "https://api.llmam3.com/generate"  # Replace with your API endpoint
llm_api_key = "YOUR_LLLAMAM3_API_KEY"  # Replace with your API key if needed
hyde_llm = CustomLLM(api_url=llm_api_url, api_key=llm_api_key, temperature=0.7)

# Query Expansion LLM (using custom LLM)
query_expansion_llm = CustomLLM(api_url=llm_api_url, api_key=llm_api_key, temperature=0.7)

# Define the function for the chatbot
def chatbot(query):
    # Step 1: Use HYDE to generate a hypothetical answer
    hyde_prompt_template = PromptTemplate(
        input_variables=["question"],
        template="Provide a detailed answer to the following question:\n{question}"
    )
    hyde_chain = LLMChain(llm=hyde_llm, prompt=hyde_prompt_template)
    hypothetical_answer = hyde_chain.run(question=query)

    # Step 2: Query Expansion to generate 5 more relevant questions
    expansion_prompt_template = PromptTemplate(
        input_variables=["question"],
        template="Generate 5 questions that are relevant to the following question:\n{question}\nQuestions:"
    )
    expansion_chain = LLMChain(llm=query_expansion_llm, prompt=expansion_prompt_template)
    expanded_questions_text = expansion_chain.run(question=query)
    expanded_questions = [q.strip() for q in expanded_questions_text.split('\n') if q.strip()]

    # Combine the original query with expanded queries
    all_queries = [query] + expanded_questions

    # Step 3: Retrieve documents from Redis for all queries
    retriever = RedisRetriever(
        redis_url=redis_url,
        index_name=VECTOR_STORE_INDEX_NAME,
        embedding_function=embedding_model.embed_query,
    )

    documents = []
    for q in all_queries:
        docs = retriever.get_relevant_documents(q)
        documents.extend(docs)

    # Remove duplicates
    unique_documents = {doc.page_content: doc for doc in documents}.values()
    documents = list(unique_documents)

    # Step 4: Rerank the documents using cross-encoder
    if documents:
        # Prepare the cross-encoder inputs
        cross_encoder_inputs = [(query, doc.page_content) for doc in documents]
        # Compute relevance scores
        relevance_scores = reranker.predict(cross_encoder_inputs)
        # Attach scores to documents
        for doc, score in zip(documents, relevance_scores):
            doc.metadata['relevance_score'] = score
        # Sort documents by score (higher is better)
        documents.sort(key=lambda doc: doc.metadata['relevance_score'], reverse=True)
        # Select top documents (you can adjust the number)
        top_documents = documents[:5]
    else:
        top_documents = []

    # Step 5: Generate the final answer using the top documents
    # Prepare context from top documents
    context = "\n\n".join([doc.page_content for doc in top_documents])

    final_prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        You are an AI assistant. Use the following context to answer the question.

        Context:
        {context}

        Question:
        {question}

        Provide the answer in bullet points:
        """
    )
    final_chain = LLMChain(llm=hyde_llm, prompt=final_prompt_template)
    final_answer = final_chain.run(question=query, context=context)

    # Ensure the answer is formatted in bullet points
    bullet_points = [line.strip() for line in final_answer.split('\n') if line.strip()]
    bullet_points = [f"- {point}" if not point.startswith('-') else point for point in bullet_points]
    final_response = '\n'.join(bullet_points)

    return final_response

# Create Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="AI RAG Chatbot",
    description="Ask any question and receive an answer based on the augmented retrieval.",
    examples=[
        ["What are the benefits of renewable energy?"],
        ["Explain the theory of relativity."],
        ["How does the stock market work?"],
    ],
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
