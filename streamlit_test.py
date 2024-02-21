import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Streamlit App",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add an image header
#st.image("your_image.png", use_container_width=True)

# Set sidebar color to emerald
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #50C878;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a form
with st.form(key="question_form"):
    question = st.text_input("Enter your question:")
    submit_button = st.form_submit_button("Submit")

# Suggested questions (you can customize these)
suggested_questions = {
    "What is Streamlit?": "https://streamlit.io",
    "How to deploy Streamlit apps?": "https://docs.streamlit.io/deploy_streamlit_app.html",
    "Streamlit community forum": "https://discuss.streamlit.io",
}

# Display suggested question links
st.sidebar.header("Suggested Questions")
for q, link in suggested_questions.items():
    if st.sidebar.button(q, key=q):
        question = q  # Populate input with clicked question

# Display user's input
st.write(f"Your question: {question}")