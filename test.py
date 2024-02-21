"""st.markdown("This text is :red[colored red], and this is **:blue[colored]** and bold.")
st.markdown(
    "<h3 style='text-align: center; color: white;'>Built by Abhijeet Rajput </a></h3>",
    unsafe_allow_html=True)
st.markdown(
    'This will print <span style="color:blue;"> blue text </span>',
    unsafe_allow_html=True
) """

import streamlit as st
from streamlit_chat import message
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_customllm import CustomLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import re

#{format_instructions}

member_api = requests.get("http://localhost:8001/user").text.replace("{", '<').replace("}", '>')

#Use only the following context to answer the question. Give very specific answer and dont give explanation when not asked.
template = """Instructions: You are Florida Blue health-care Customer Service chatbot tasked to help answer the member question with right information.
Agent will use availabe context only to provide information to member and provide concise and to the point answer, without giving 
additonal explaination unless asked. Please avoid irrelevant information or adding policy information when not needed.
Your response should be a information passage and list of steps eg: 1. , 2., 3. etc.
example : 
Question : what are member copays?
answer: member copays are as below 
1. $10 of the PCP copay
2. $40 for Specialist copay

Question: {question}
Context: {context} and """ + f""" member policy details are as below : {member_api}

answer:
"""
# all the variables specific to RAG LLM - Retrieval-Augmented Generation
embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
llm = CustomLLM()
db = Chroma(persist_directory="./chromadb_flblue3", embedding_function=embedder)

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
    #st.session_state['contextArea'] = [context]
    #st.session_state['contextArea'].append(context)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"],partial_variables={"format_instructions": format_instructions},)
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 1}),
                                          # memory=memory,
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt},
                                           )
    #ConversationalRetrievalChain.from_llm(llm=llm,)
    #ConversationalRetrievalChain(chain=qa_chain, memory=memory, output_parser=output_parser)
    response = qa_chain({'query': question})
    answer = response['result'] #[0]['answer']
    #st.session_state['history'].append((query, answer))
    return answer

response = conversational_chat("Summarize my policy?") #"how to use health insurance marketplace?"
pattern = r'"answer": "(.*?)"'
match = re.search(pattern, response)
#extracted_answer = match.group(1)
print(response)

"""
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
st.title("HealthPal")
st.caption("Please enter your questions to get information about your policy or health-care at florid Blue")

# Suggested questions (you can customize these)
suggested_questions = {
    "Can you Summarize my policy ?": "Can you Summarize my policy ?",
    "Give me information related to discounts and rewards": "Give me information related to discounts and rewards",
    "How to Save on Imaging Services?": "How to Save on Imaging Services?",
    "Give me details about copays on my policy": "Give me details about copays on my policy",
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

"""