
import streamlit as st
from streamlit_chat import message
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_customllm_hf import CustomLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.memory import ChatMessageHistory, ConversationBufferMemory


#member_api = requests.get("http://localhost:8001/user").text.replace("{", '<').replace("}", '>')
#{format_instructions}
template = """Answer the question based only on the  context only, if information not available in context then mention "no information available".

### Output ###
if the question is related to medication or drug information, please provide the response in bullet points with 
details about drug name, tier, speciality, wheather covered or not, requirement/limits.
example : description of drugs
Tier
Speciality
Requirement/Limit
Covered

### VERIFICATION ###
 Is the above summary accurate based on the original context?
 
 
for all the responses, format the response in clear and bullet point summary.Make sure information is based on context, refining for accuracy.

Question: {question}
Context: {context}

answer:
"""
# all the variables specific to RAG LLM - Retrieval-Augmen`ted Generation
embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
llm = CustomLLM()
db = Chroma(persist_directory="./chromadb_uwib7", embedding_function=embedder)

response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
#output_parser = StructuredOutputParser(output_schema=ResponseSchema(output_format='text'))
#print(output_parser.get_format_instructions())
format_instructions = output_parser.get_format_instructions()

#prompt = PromptTemplate(template=template, input_variables=["context", "question"])
# Setup memory for contextual conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def conversational_chat(query):
    print(query)
    question = query
    prompt = PromptTemplate(template=template, input_variables=["context", "question"],partial_variables={"format_instructions": format_instructions},)
    retriever = db.as_retriever(search_kwargs={'k': 4})
    print(retriever)
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=retriever, #db.as_retriever(search_kwargs={'k': 1}),
                                          # memory=memory,
                                           return_source_documents=True,
                                           chain_type_kwargs={"verbose": True,'prompt': prompt},
                                           verbose=True
                                           )
    #ConversationalRetrievalChain.from_llm(llm=llm,)
    #ConversationalRetrievalChain(chain=qa_chain, memory=memory, output_parser=output_parser)
    response = qa_chain({'query': question})
    answer = response['result'] #[0]['answer']
    return answer

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! I can help you with your health-care questions. Ask me anything !"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

# Set custom theme colors
st.set_page_config(
    page_title="Customer Support Agent",
    page_icon=":gem:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add header image
st.image("https://www.freewebheaders.com/wp-content/gallery/other-business/call-centre-agents-with-headsets-website-header.jpg", use_column_width = "auto")

# Add sidebar content
st.sidebar.title("More Suggestions for You")

# Add a text input for letting member input the question
st.title("UWIB Agent")
st.caption("Please enter your questions to get information about your policy or health-care at florida Blue")

# Suggested questions (you can customize these)
suggested_questions = {
    "Can you Summarize my policy ?": "Can you Summarize my policy ?",
    "Give me information related to discounts and rewards": "Give me information related to discounts and rewards",
    "How to Save on Imaging Services?": "How to Save on Imaging Services?",
    "How to make payment at florida blue?": "How to make payment at florida blue?",
    "How to to use healthcare marketplace?": "How to to use healthcare marketplace?",
}

# Initialize session state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Display suggested question links
#';st.sidebar.header("Suggested Questions")
for q, link in suggested_questions.items():
    if st.sidebar.button(q, key=q):
        question = q  # Populate input with clicked question
        st.session_state.user_input = q

# container for the chat history
response_container = st.container()
# container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", value= st.session_state.user_input, placeholder="Enter your questions here...", key='input')
        st.empty()
        submit_button =  st.form_submit_button(label='Submit')
        st.empty()

    if submit_button and user_input:
        output = conversational_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['user_input'] = ""

if st.session_state['generated']:
   with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="no-avatar")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")