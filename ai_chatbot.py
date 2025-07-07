# ai_chatbot.py

import streamlit as st
from transformers import pipeline

# -----------------------------
#  Mental Health Support Chatbot – Flan-T5 Final Version with Larger Font
# -----------------------------

# Page configuration
st.set_page_config(page_title="Mental Health Chatbot", page_icon="", layout="wide")

# Title
st.title(" Mental Health Support Chatbot ")
st.write("This chatbot uses Flan-T5 to provide empathetic mental health responses. Avoid sharing personal sensitive data.")

# Load Flan-T5 model with caching
@st.cache_resource()
def load_flan_t5():
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")
    return qa_pipeline

qa_pipeline = load_flan_t5()

# Initialise session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history with larger, clear font
for user_msg, bot_msg in st.session_state.messages:
    st.markdown(
        f"""
        <div style='
            background-color:#d4edda;
            color:#155724;
            padding:10px;
            border-radius:8px;
            margin:5px 0;
            text-align:right;
            font-size:16px;'>
            {user_msg}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div style='
            background-color:#f8f9fa;
            color:#212529;
            padding:10px;
            border-radius:8px;
            margin:5px 0;
            text-align:left;
            font-size:16px;'>
            {bot_msg}
        </div>
        """,
        unsafe_allow_html=True
    )

# Input box and send button
user_input = st.text_input("Type your message here:")

if st.button("Send"):
    if user_input:
        # Prompt with empathetic instruction
        prompt = f"Provide an empathetic mental health support response to: {user_input}"

        # Generate output
        output = qa_pipeline(prompt, max_length=150, do_sample=True)[0]['generated_text']

        # Update session state with new message
        st.session_state.messages.append((user_input, output))

# Mental health resources
with st.expander("🌐 Mental Health Resources"):
    st.markdown("""
    - [iCall India Helpline](https://icallhelpline.org)
    - [AASRA (India Suicide Prevention)](http://www.aasra.info)
    - [WHO Mental Health Resources](https://www.who.int/mental_health/en/)
    """)
