import os
import gradio as gr
import requests
import json
from typing import List, Dict, Any, Optional, Callable, Tuple

import numpy as np
from langchain.schema import Document, BaseRetriever
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.manager import CallbackManagerForLLMRun

# Custom Embedder that calls an external API
class CustomEmbedder(Embeddings):
    def __init__(self, api_url: str, api_key: str, dimensions: int = 768):
        self.api_url = api_url
        self.api_key = api_key
        self.dimensions = dimensions
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the API"""
        payload = {"texts": texts}
        response = requests.post(
            f"{self.api_url}/embed-documents", 
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise ValueError(f"API error: {response.text}")
        
        return response.json()["embeddings"]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the API"""
        payload = {"text": text}
        response = requests.post(
            f"{self.api_url}/embed-query", 
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise ValueError(f"API error: {response.text}")
        
        return response.json()["embedding"]

# Custom LLM that calls an external API
class CustomLLM(LLM):
    def __init__(self, api_url: str, api_key: str, model_name: str = "default"):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the API to generate text"""
        payload = {
            "prompt": prompt,
            "model": self.model_name,
            "stop": stop if stop else []
        }
        payload.update(kwargs)
        
        response = requests.post(
            f"{self.api_url}/generate", 
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise ValueError(f"API error: {response.text}")
        
        return response.json()["text"]
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM"""
        return "custom_api"

# Query Expansion using LLM
class QueryExpander:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Generate 3 different versions of the following query. 
            These should cover different aspects and potential interpretations of the query.
            Make the expanded queries diverse but relevant.
            
            Original query: {query}
            
            Expanded queries (provide exactly 3, one per line):
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def expand_query(self, query: str) -> List[str]:
        """Expand a query into multiple related queries"""
        result = self.chain.run(query)
        expanded_queries = [line.strip() for line in result.split('\n') if line.strip()]
        # Ensure we always include the original query
        return [query] + expanded_queries[:3]

# HYDE (Hypothetical Document Embeddings) Implementation
class HYDERetriever(BaseRetriever):
    def __init__(self, llm: LLM, embedder: Embeddings, vectorstore, top_k: int = 5):
        super().__init__()
        self.llm = llm
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.hyde_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are an expert research assistant. Your task is to write a detailed passage 
            that would serve as a perfect answer to the given query.
            
            Query: {query}
            
            Write a detailed passage that answers this query:
            """
        )
        self.hyde_chain = LLMChain(llm=self.llm, prompt=self.hyde_prompt)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Generate a hypothetical document and use it for retrieval"""
        # Generate a hypothetical document
        hypothetical_doc = self.hyde_chain.run(query)
        
        # Use the hypothetical document as a query for similarity search
        docs = self.vectorstore.similarity_search_by_vector(
            self.embedder.embed_query(hypothetical_doc),
            k=self.top_k
        )
        
        return docs

# Embedding-based Reranker
class EmbeddingReranker:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
    
    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """Rerank documents based on embedding similarity"""
        if not documents:
            return []
        
        # Embed the query
        query_embedding = self.embeddings.embed_query(query)
        
        # Calculate similarity scores
        doc_scores = []
        for doc in documents:
            doc_embedding = self.embeddings.embed_documents([doc.page_content])[0]
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            doc_scores.append((doc, similarity))
        
        # Sort by similarity (descending)
        ranked_docs = [doc for doc, _ in sorted(doc_scores, key=lambda x: x[1], reverse=True)]
        
        # Return top N
        return ranked_docs[:top_n]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
        
        return dot_product / (norm_vec1 * norm_vec2)

# Advanced RAG System
class AdvancedRAG:
    def __init__(
        self, 
        llm: LLM, 
        embedder: Embeddings,
        documents_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.llm = llm
        self.embedder = embedder
        
        # Load and process documents
        self.documents = self._load_documents(documents_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.splits = self.text_splitter.split_documents(self.documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=self.splits,
            embedding=self.embedder
        )
        
        # Initialize components
        self.query_expander = QueryExpander(llm)
        self.hyde_retriever = HYDERetriever(llm, embedder, self.vectorstore)
        self.reranker = EmbeddingReranker(embedder)
        
        # QA Chain for generating the final answer
        self.qa_chain = load_qa_chain(
            llm=llm,
            chain_type="stuff",
            prompt=PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are a helpful assistant providing accurate information.
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer the question based on the context provided. If the context doesn't contain relevant information,
                say so and provide what you know about the topic.
                """
            )
        )
    
    def _load_documents(self, path: str) -> List[Document]:
        """Load documents from a directory or file"""
        # This is a simplified placeholder for document loading
        # In a real implementation, you would handle different file types
        # and use appropriate document loaders
        
        # For this example, we'll just create a dummy document
        return [Document(page_content="This is a placeholder document.")]
    
    def process_query(self, query: str, use_query_expansion: bool = True) -> str:
        """Process a query through the full RAG pipeline"""
        # Step 1: Query expansion (optional)
        if use_query_expansion:
            expanded_queries = self.query_expander.expand_query(query)
        else:
            expanded_queries = [query]
        
        # Step 2: Retrieve documents using HYDE for each expanded query
        all_docs = []
        for exp_query in expanded_queries:
            docs = self.hyde_retriever.get_relevant_documents(exp_query)
            all_docs.extend(docs)
        
        # Remove duplicates (if any)
        unique_docs = []
        seen_contents = set()
        for doc in all_docs:
            if doc.page_content not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(doc.page_content)
        
        # Step 3: Rerank documents using embedding similarity
        reranked_docs = self.reranker.rerank(query, unique_docs, top_n=5)
        
        # Step 4: Generate response using LLM
        if reranked_docs:
            context = "\n\n".join([doc.page_content for doc in reranked_docs])
            response = self.qa_chain.run(context=context, question=query)
        else:
            response = "I couldn't find any relevant information to answer your question."
        
        return response

# Gradio Interface
def create_gradio_interface(rag_system: AdvancedRAG):
    """Create a Gradio interface for the RAG system"""
    
    def chat_response(query, history):
        """Generate a response for the Gradio chatbot"""
        response = rag_system.process_query(query)
        return response
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question")
    clear = gr.Button("Clear")
    
    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    def bot(history):
        user_message = history[-1][0]
        bot_message = chat_response(user_message, history)
        history[-1][1] = bot_message
        return history
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
    return gr.Interface(
        fn=None,
        inputs=None,
        outputs=None,
        title="Advanced RAG Chatbot",
        description="Ask questions and get answers based on a knowledge base using advanced RAG techniques.",
        article="This chatbot uses query expansion, HYDE, and embedding-based reranking.",
        blocks=[chatbot, msg, clear]
    )

# Main function
def main():
    # API Configuration (replace with your actual API endpoints and keys)
    embedding_api_url = "https://your-embedding-api-endpoint.com"
    embedding_api_key = os.environ.get("EMBEDDING_API_KEY", "your-embedding-api-key")
    
    llm_api_url = "https://your-llm-api-endpoint.com"
    llm_api_key = os.environ.get("LLM_API_KEY", "your-llm-api-key")
    
    # Initialize custom components
    embedder = CustomEmbedder(embedding_api_url, embedding_api_key)
    llm = CustomLLM(llm_api_url, llm_api_key)
    
    # Initialize RAG system
    rag_system = AdvancedRAG(
        llm=llm,
        embedder=embedder,
        documents_path="./documents/",  # Replace with your documents path
    )
    
    # Create and launch Gradio interface
    interface = create_gradio_interface(rag_system)
    interface.launch(share=True)

if __name__ == "__main__":
    main()
