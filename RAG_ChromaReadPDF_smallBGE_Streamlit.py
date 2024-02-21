import streamlit as st
from streamlit_chat import message
import tempfile
import os
import requests
import chromadb
from embedder_smallBGE import CustomEmbedder

class RAGPipeline():
    def __init__(self) -> None:
        self.API_TOKEN = "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" #os.environ["API_TOKEN"]
        self.API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"

    def query(self, prompt):
        headers = {"Authorization": f"Bearer {self.API_TOKEN}"}

        payload = {
            "inputs": prompt,
            "parameters": { #Try and experiment with the parameters
                "max_new_tokens": 1000,
                "temperature": 0.5  ,
                "top_p": 0.9,
                "do_sample": False,
                "return_full_text": False
            }
        }
        response = requests.post(self.API_URL, headers=headers, json=payload)
        return response.json()[0]['generated_text']

    def as_retriever(self, collection, query):
        results = collection.query(
            query_texts=[query],
            n_results=1
        )
        return results['documents']

# Create an instance of the class and initialize the variables
rag_app = RAGPipeline()
custom_embedder = CustomEmbedder()
client = chromadb.PersistentClient(path="../mydb_smallBGE")
collection = client.get_collection(name="rag_collection_smallBGE", embedding_function=custom_embedder)


def conversational_chat(query):
    print(query)
    question = query
    context = rag_app.as_retriever(collection, question)
    st.session_state['contextArea'] = [context]
    #st.session_state['contextArea'].append(context)
    prompt = f"""Use the following context to answer the question at the end. Stop when you've answered the question. Do not generate any more than that.

    {context}

    Question: {question}
    """
    answer = rag_app.query(prompt)
    st.session_state['history'].append((query, answer))
    return answer

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

if 'contextArea' not in st.session_state:
    st.session_state['contextArea'] = ["nothing yet"]

st.title("Chat with CSV using Zephyr")
st.markdown(
    "<h3 style='text-align: center; color: white;'>Built by Abhijeet Rajput </a></h3>",
    unsafe_allow_html=True)

# container for the chat history
response_container = st.container()
# container for the user's text input
container = st.container()
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk to LLM here .... :)", key='input')
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

st.sidebar.write("Context Area:", st.session_state['contextArea'])
