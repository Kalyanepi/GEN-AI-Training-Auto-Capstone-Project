# streamlit_app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
load_dotenv(dotenv_path="/home/ubuntu/insurance_lab/.env")

# ■■ Page Configuration ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
st.set_page_config(
page_title="InsureSafe AI Assistant",
page_icon="■■",
layout="wide"
)

# ■■ Sidebar ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
with st.sidebar:
    st.title("■■ Settings")
    model = st.selectbox("Select Model",["gpt-4o-mini", "gpt-4o"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
    "Total Messages: X"
    st.divider()
    if st.button("■■ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("**Download"):
        st.session_state.messages = []
        st.rerun()

# ■■ Main Chat Area ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
st.title("■■ InsureSafe AI Assistant")
st.caption("Ask me anything about insurance policies, claims, and coverage")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [{
    "role": "assistant",
    "content": "Hello! I'm your InsureSafe advisor. How can I help you today?"
    }]

# Display all previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if user_input := st.chat_input("Type your insurance question here..."):
# Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
with st.chat_message("user"):
    st.markdown(user_input)

# Build message history for LLM
llm = ChatOpenAI(model=model, temperature=temperature,
openai_api_key=os.getenv("OPENAI_API_KEY"))

lc_messages = [HumanMessage(content="You are an expert insurance advisor.")]
for m in st.session_state.messages:
    if m["role"] == "user":
        lc_messages.append(HumanMessage(content=m["content"]))
    else:
        lc_messages.append(AIMessage(content=m["content"]))

# Stream response to UI
with st.chat_message("assistant"):
    response = st.write_stream(llm.stream(lc_messages))
    st.session_state.messages.append({
    "role": "assistant", "content": response
    })