import streamlit as st
from llm import query_llm
import chromadb
import os
import huggingface_embedding as hf
import utils as h
from sentence_transformers import SentenceTransformer
import csv
import uuid
# import injest
import image as Image

api_url = "https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud"
api_key = "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"

client = chromadb.PersistentClient(path="./db")
collection = client.get_collection(name="vector_db")

# if "chroma_db" not in st.session_state:
#     st.session_state.chroma_db = chromadb_client.load_file()


st.title("Ask Florida Blue")

# st.sidebar.title("Upload a file")


st.markdown(
    h.add_logo_str(),
    unsafe_allow_html=True,
)

st.sidebar.image('florida-blue-logo_0.jpg', caption='')

# user_context = st.sidebar.text_area("Enter text:")
# uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt","csv"])


if 'chat-history' not in st.session_state:
    st.session_state['chat-history'] = [
        {
            "role": "ai",
            "message": "How can I help you today?"
        }

    ]

user_input = st.chat_input("Message:")

results = collection.query(query_texts=[str(user_input)], n_results=1)
print(str(results))

top_result_text = str(results['documents'][0][0])
print(top_result_text)

if user_input:
    prompt = f"<|system|>You're an assistant answering user's question. Answer the users question Only using the context given. Context: {top_result_text}</s><|user|>{user_input}<|assistant|>answer:"

    st.session_state['chat-history'].append({
        "role": "user",
        "message": user_input
    })

    llm_response = query_llm(prompt)
    # hugging_face = hf.HuggingFaceEmbeddingInference(os.environ['api_url'],os.environ['api_key'])
    # hugging_face = hf.HuggingFaceEmbeddingInference(api_url,api_key)
    # llm_response = hugging_face(prompt)

    st.session_state['chat-history'].append({
        "role": "ai",
        "message": llm_response[0]["generated_text"]
    })

    if 'chat-history' in st.session_state:
        for i in range(0, len(st.session_state['chat-history'])):
            msg = st.session_state['chat-history'][i]
            st.chat_message(msg['role']).write(msg['message'])