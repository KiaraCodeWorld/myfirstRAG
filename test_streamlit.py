import streamlit as st
from streamlit_chat import message
import tempfile
import os
import requests
import chromadb
from embedder import CustomEmbedder

"""def conversational_chat(query):
    result = { "answer" : query }
    return result["answer"]
"""

def conversational_chat(query):
    result = { "answer": query }
    st.session_state['history'].append((query,result['answer'] ))
    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

st.title("Chat with CSV using Zephyr")
st.markdown(
    "<h3 style='text-align: center; color: white;'>Built by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>",
    unsafe_allow_html=True)

# container for the chat history
response_container = st.container()
# container for the user's text input
container = st.container()
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk to LLM here .... (:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
   with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
