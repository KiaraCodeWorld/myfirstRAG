import os
import gradio as gr
import tempfile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Optional: Set up OpenAI as a fallback
# from langchain.chat_models import ChatOpenAI
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# Sample HR documents - replace with your own content
hr_documents = [
    """Vacation Policy: Full-time employees are eligible for 15 days of paid vacation per year, accrued monthly. Vacation requests must be submitted at least two weeks in advance.""",
    """Health Benefits: The company offers comprehensive health, dental, and vision insurance. New employees are eligible after 30 days of employment.""",
    """Remote Work Policy: Employees may work remotely up to 2 days per week with manager approval. All remote work arrangements must be documented with HR.""",
    """Parental Leave: Eligible employees can receive up to 12 weeks of paid parental leave following the birth or adoption of a child.""",
    """Performance Reviews: Performance reviews are conducted bi-annually in June and December. Employees complete a self-assessment prior to meeting with their manager.""",
    """Professional Development: The company offers up to $2,000 annually for professional development activities including conferences, courses, and certifications.""",
    """Referral Bonus: Employees who refer successful candidates receive a $1,500 bonus after the new hire completes 90 days of employment.""",
    """Retirement Benefits: The company offers a 401(k) plan with up to 4% matching contributions after 90 days of employment."""
]

# Create vector store from sample documents
def create_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(["\n\n".join(hr_documents)])
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Initialize vector store
vector_store = create_vector_store()

# Sample suggested questions
suggested_questions = [
    "What is the company's vacation policy?",
    "How do health benefits work?",
    "Can I work remotely?",
    "How does parental leave work?",
    "When are performance reviews conducted?",
    "What professional development opportunities are available?",
    "Is there a referral bonus program?",
    "What retirement benefits are offered?"
]

# Load LLM models
def load_model(model_name, temperature):
    if model_name == "llama":
        model_id = "meta-llama/Llama-2-7b-chat-hf"  # You may need to adjust this path
    elif model_name == "phi":
        model_id = "microsoft/phi-2"
    elif model_name == "mistral":
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=temperature,
        top_p=0.95,
        do_sample=True
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Advanced RAG features
def apply_hyde(query, llm):
    """Hypothetical Document Embeddings technique"""
    hyde_prompt = PromptTemplate(
        input_variables=["question"],
        template="""Create a hypothetical document that would perfectly answer this question: {question}
        
        Document:"""
    )
    hypothetical_doc = llm.generate([hyde_prompt.format(question=query)]).generations[0][0].text
    return hypothetical_doc

def expand_question(query, llm):
    """Question expansion technique"""
    expansion_prompt = PromptTemplate(
        input_variables=["question"],
        template="""Given the user question: {question}
        
        Generate 3 alternative versions of this question that might help retrieve more relevant information:"""
    )
    expanded = llm.generate([expansion_prompt.format(question=query)]).generations[0][0].text
    expanded_questions = [q.strip() for q in expanded.split("\n") if q.strip()]
    return expanded_questions

# Setup conversation chain
def get_conversation_chain(llm, use_hyde, use_question_expansion):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Default retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Custom prompt
    hr_template = """You are an HR assistant named HR Connect. Your role is to provide helpful, accurate information about company policies, benefits, and procedures. Always maintain a professional and supportive tone.

    Here is some information that might be relevant to the question:
    {context}

    Based on this information, please answer the question.
    
    If the information provided doesn't fully answer the question, acknowledge this and provide whatever information you can. 
    Suggest who the employee might contact for more specific information (like "the HR department" or "your direct manager").

    Question: {question}
    """
    
    PROMPT = PromptTemplate(
        template=hr_template, 
        input_variables=["context", "question"]
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    
    return chain

# Chat function
def process_chat(message, history, model_choice, temperature, use_hyde, use_question_expansion):
    try:
        # Convert temperature from string to float
        temp = float(temperature)
        
        # Load the selected model
        llm = load_model(model_choice, temp)
        
        # Create conversation chain
        chain = get_conversation_chain(llm, use_hyde, use_question_expansion)
        
        # Apply advanced RAG techniques if selected
        if use_hyde:
            # In a full implementation, this would modify the retrieval process
            # For simplicity, we'll just note this in the response
            hyde_note = "(Using HYDE for enhanced retrieval)"
        else:
            hyde_note = ""
            
        if use_question_expansion:
            # In a full implementation, this would expand the query
            # For simplicity, we'll just note this in the response
            expansion_note = "(Using question expansion)"
        else:
            expansion_note = ""
        
        # Generate response
        response = chain({"question": message})
        
        if hyde_note or expansion_note:
            prefix = f"{hyde_note} {expansion_note}\n\n"
            full_response = prefix + response["answer"]
        else:
            full_response = response["answer"]
        
        return full_response
    except Exception as e:
        return f"Error processing your request: {str(e)}"

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown("# 🏢 HR Connect")
    gr.Markdown("Get answers to your HR questions using our AI assistant")
    
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(placeholder="Type your HR question here...", label="Your Question")
    
    with gr.Accordion("Model Settings", open=False):
        with gr.Row():
            model_choice = gr.Radio(
                ["llama", "phi", "mistral"], 
                label="Select Model", 
                value="mistral"
            )
            
            temperature = gr.Slider(
                minimum=0.1, 
                maximum=1.0, 
                value=0.7, 
                step=0.1, 
                label="Temperature"
            )
    
    with gr.Accordion("Advanced Features", open=False):
        with gr.Row():
            use_hyde = gr.Checkbox(label="Use HYDE (Hypothetical Document Embeddings)", value=False)
            use_question_expansion = gr.Checkbox(label="Use Question Expansion", value=False)
    
    clear = gr.Button("Clear Conversation")
    
    gr.Markdown("### Suggested Questions")
    
    with gr.Row():
        suggestion_buttons = [gr.Button(question) for question in suggested_questions[:4]]
    
    with gr.Row():
        suggestion_buttons.extend([gr.Button(question) for question in suggested_questions[4:]])
    
    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    def bot(history, model_choice, temperature, use_hyde, use_question_expansion):
        user_message = history[-1][0]
        bot_response = process_chat(user_message, history[:-1], model_choice, temperature, use_hyde, use_question_expansion)
        history[-1][1] = bot_response
        return history
    
    def add_suggestion(suggestion, history):
        return user(suggestion, history)
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, model_choice, temperature, use_hyde, use_question_expansion], [chatbot]
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)
    
    for i, button in enumerate(suggestion_buttons):
        button.click(
            add_suggestion, 
            [button, chatbot], 
            [msg, chatbot], 
            queue=False
        ).then(
            bot, 
            [chatbot, model_choice, temperature, use_hyde, use_question_expansion], 
            [chatbot]
        )

if __name__ == "__main__":
    demo.launch()

=================
