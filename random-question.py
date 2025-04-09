from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

def generate_starter_questions(vector_store, num_questions=3):
    """
    Generate potential starter questions from vector store content
    """
    # 1. Get representative samples from vector store
    sample_docs = vector_store.similarity_search("key concepts", k=5)
    
    # 2. Create question generation prompt
    prompt_template = PromptTemplate(
        input_variables=["documents"],
        template="Generate {num_questions} concise starter questions based on these documents:\n"
                "{documents}\n\n"
                "Format as numbered list without explanations."
    )
    
    # 3. Configure question generation chain
    llm = OpenAI(temperature=0.7)
    question_chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )
    
    # 4. Generate and return questions
    documents_text = "\n".join([doc.page_content for doc in sample_docs])
    return question_chain.run(
        documents=documents_text,
        num_questions=num_questions
    )

# Usage Example
if __name__ == "__main__":
    # Initialize vector store (replace with your actual store)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("my_vector_store", embeddings)
    
    # Generate 3 starter questions
    questions = generate_starter_questions(vector_store, 3)
    print("Suggested starter questions:")
    print(questions)

========= deepseek:

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI  # Can be replaced with other LLMs

def generate_headstart_questions(vector_store, query, llm=None, num_questions=5, k=3):
    """
    Generates relevant questions to explore a topic using vector store context.
    
    Args:
        vector_store: Initialized vector store instance
        query: Topic/question seed for the headstart
        llm: LangChain LLM instance (default: OpenAI with temperature=0.7)
        num_questions: Number of questions to generate
        k: Number of documents to retrieve from vector store
    
    Returns:
        List of generated questions
    """
    # Set default LLM if not provided
    if llm is None:
        llm = OpenAI(temperature=0.7)
    
    # 1. Retrieve relevant documents from vector store
    documents = vector_store.similarity_search(query, k=k)
    
    if not documents:
        return ["No relevant information found to generate questions."]
    
    # 2. Create context from retrieved documents
    context = "\n".join([doc.page_content for doc in documents])
    
    # 3. Create question generation prompt template
    prompt_template = """Based on the following context information, generate {num_questions} 
    insightful questions that would help someone explore this topic. The questions should:
    - Be open-ended
    - Cover different aspects of the topic
    - Address potential complexities
    - Encourage deeper investigation
    
    Context:
    {context}
    
    Generate {num_questions} questions that follow these requirements:""".replace("    ", "")
    
    prompt = PromptTemplate(
        input_variables=["context", "num_questions"],
        template=prompt_template
    )
    
    # 4. Create and run LLM chain
    question_chain = LLMChain(llm=llm, prompt=prompt)
    generated_questions = question_chain.run(
        context=context,
        num_questions=num_questions
    )
    
    # 5. Process and return results
    questions = [q.strip() for q in generated_questions.split("\n") if q.strip()]
    return questions[:num_questions]

# Example usage
if __name__ == "__main__":
    # Initialize your vector store (example using Chroma)
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings
    
    # Replace with your actual vector store initialization
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(persist_directory="./your_vector_store", embedding_function=embeddings)
    
    # Generate questions about machine learning
    questions = generate_headstart_questions(
        vector_store=vector_store,
        query="machine learning fundamentals",
        num_questions=5,
        k=4  # Retrieve 4 documents for context
    )
    
    print("Suggested exploration questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

======== 
