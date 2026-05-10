
# full_chatbot.py
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv(dotenv_path="/home/ubuntu/insurance_lab/.env")

st.set_page_config(
    page_title="InsureSafe Pro",
    page_icon="■",
    layout="wide"
)

# ---------------------------
# Session State Initialization
# ---------------------------
def init_session():
    defaults = {
        "messages": [],
        "policy_context": "",
        "policy_name": None,
        "doc_loaded": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()

# -----------------------------------------
# Extract policy text (cached for same file)
# -----------------------------------------
@st.cache_data(show_spinner=False)
def extract_policy_text(pdf_bytes: bytes, filename: str) -> str:
    """Extract text from PDF using Docling and return markdown text."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        text = result.document.export_to_markdown()
        return text
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ---------------------------
# Sidebar: Upload + Settings
# ---------------------------
with st.sidebar:
    st.header("■ Upload Policy Document")

    uploaded = st.file_uploader(
        "Upload an insurance policy PDF",
        type=["pdf"],
        help="The chatbot will answer questions based on this document"
    )

    st.divider()

    # Always define model and temperature (so variables exist even without doc)
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)

    st.divider()

    if st.button("■ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------
# Handle new upload
# ---------------------------
if uploaded and uploaded.name != st.session_state.policy_name:
    with st.spinner("■ Reading policy document..."):
        pdf_text = extract_policy_text(uploaded.read(), uploaded.name)
        st.session_state.policy_context = pdf_text
        st.session_state.policy_name = uploaded.name
        st.session_state.doc_loaded = True
        st.session_state.messages = []
    st.success(f"■ Loaded: {uploaded.name}")

# ---------------------------
# Layout
# ---------------------------
col1, col2 = st.columns([3, 1])

# ===========================
# Main Chat Area
# ===========================
with col1:
    st.title("■ InsureSafe Pro Advisor")

    # System prompt grounded in uploaded document
    if st.session_state.doc_loaded:
        system_msg = SystemMessage(
            content=(
                "You are an expert insurance advisor. Use ONLY the policy document below "
                "to answer questions. If something is not in the document, say so.\n\n"
                f"POLICY DOCUMENT:\n{st.session_state.policy_context[:6000]}"
            )
        )
        st.caption(f"■ Grounded on: {st.session_state.policy_name}")
        st.info(f"■ Active: {st.session_state.policy_name}")
    else:
        system_msg = SystemMessage(
            content="You are an expert insurance advisor. Answer general insurance questions."
        )
        st.caption("■ No document uploaded — answering from general knowledge")
        st.warning("Upload a policy PDF for document-grounded answers.")

    st.divider()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input should be OUTSIDE the history loop
    user_input = st.chat_input("Ask about this policy...")

    if user_input:
        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Build LangChain message list with history
        lc_msgs = [system_msg]
        for m in st.session_state.messages:
            msg_cls = HumanMessage if m["role"] == "user" else AIMessage
            lc_msgs.append(msg_cls(content=m["content"]))