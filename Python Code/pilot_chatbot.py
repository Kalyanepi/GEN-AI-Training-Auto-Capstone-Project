
# full_chatbot.py
import os
import tempfile
import sys
import time
import json

import streamlit as st
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Shared corpus
sys.path.insert(0, "/home/ubuntu/insurance_lab/shared")
from corpus import CORPUS


# Load environment variables
load_dotenv(dotenv_path="/home/ubuntu/insurance_lab/.env")

st.set_page_config(
    page_title="Insure Pro",
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
    policy_type = st.selectbox("policy type", ["All","health","motor","life"], index=1)
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
    st.title("■ Insure Pro Advisor")

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

# =========================================================
# OPENAI EMBEDDINGS
# =========================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# =========================================================
# LOAD EXISTING FAISS INDEX
# =========================================================

IDX = "../lab9/faiss_insurance_index"

if os.path.exists(IDX):

    vs = FAISS.load_local(
        IDX,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("Loaded index from Lab 9 cache")

else:

    print("Lab 9 index not found. Building new index...")

    vs = FAISS.from_documents(CORPUS, embeddings)

    vs.save_local(IDX)

    print("New FAISS index built and saved")


# =========================================================
# FILTERED SEARCH FUNCTION
# =========================================================

def filtered_search(query, filters=None, k=3):

    """
    Semantic search with optional metadata filters.

    Example:
        {"policy_type": "health"}

    or:
        {"policy_type": "motor", "section": "add_ons"}
    """

    t0 = time.time()

    # With metadata filter
    if filters:

        results = vs.similarity_search_with_score(
            query,
            k=k,
            filter=filters
        )

    # Without filter
    else:

        results = vs.similarity_search_with_score(
            query,
            k=k
        )

    latency = round((time.time() - t0) * 1000, 1)

    tag = f"FILTERED {filters}" if filters else "UNFILTERED"

    print("\n" + "=" * 80)
    print(f"[{tag}]")
    print("=" * 80)

    print(f"Query: {query}")
    print(f"Latency: {latency} ms")
    print(f"Results Returned: {len(results)}")

    print("-" * 80)

    for rank, (doc, score) in enumerate(results, start=1):

        m = doc.metadata

        print(
            f"{rank}. "
            f"[{m['doc_id']}] "
            f"{m['policy_type']}/{m['claim_type']} "
            f"score={score:.4f}"
        )

        print(f"   section = {m['section']}")
        print(f"   {doc.page_content[:120]}...")
        print()

    return results, latency


# =========================================================
# SMART AUTO FILTER
# =========================================================

def smart_search(query, k=3):

    """
    Auto infer policy_type from query text
    """

    q = query.lower()

    filters = None

    # Motor-related queries
    if any(word in q for word in [
        "motor",
        "car",
        "vehicle",
        "garage",
        "engine",
        "idv",
        "accident"
    ]):

        filters = {"policy_type": "motor"}

    # Life-related queries
    elif any(word in q for word in [
        "life",
        "death",
        "nominee",
        "premium waiver",
        "surrender",
        "policyholder"
    ]):

        filters = {"policy_type": "life"}

    # Health-related queries
    elif any(word in q for word in [
        "hospital",
        "surgery",
        "cashless",
        "diabetes",
        "maternity",
        "health"
    ]):

        filters = {"policy_type": "health"}

    return filtered_search(query, filters, k)


# =========================================================
# MAIN
# =========================================================

def main():

    experiments = [

        {
            "label": "Unfiltered baseline",
            "query": "Is cosmetic surgery covered?",
            "filter": None
        },

        {
            "label": "Health-only filter",
            "query": "Is cosmetic surgery covered?",
            "filter": {"policy_type": "health"}
        },

        {
            "label": "Motor claims procedure",
            "query": "How do I register a vehicle damage claim?",
            "filter": {
                "policy_type": "motor",
                "section": "claims_procedure"
            }
        },

        {
            "label": "Life riders only",
            "query": "What extra coverage can I add to my life policy?",
            "filter": {
                "policy_type": "life",
                "section": "riders"
            }
        },

        {
            "label": "Motor add-ons monsoon",
            "query": "What add-ons protect my car in monsoon floods?",
            "filter": {
                "policy_type": "motor",
                "section": "add_ons"
            }
        },

        {
            "label": "Health exclusion dual-filter",
            "query": "What is not covered under health insurance?",
            "filter": {
                "policy_type": "health",
                "claim_type": "exclusion"
            }
        }
    ]

    records = []

    # Run experiments
    for exp in experiments:

        print("\n" + "#" * 90)
        print(f"EXPERIMENT: {exp['label']}")
        print("#" * 90)

        results, latency = filtered_search(
            exp["query"],
            exp["filter"]
        )

        records.append({
            "label": exp["label"],
            "query": exp["query"],
            "filter": str(exp["filter"]),
            "latency_ms": latency,
            "results_returned": len(results)
        })

    # =====================================================
    # SMART SEARCH DEMOS
    # =====================================================

    print("\n" + "=" * 90)
    print("SMART SEARCH DEMOS")
    print("=" * 90)

    smart_queries = [

        "When does my car NCB reset?",

        "When does NCB reset?",

        "How to file hospital cashless claim?",

        "How nominee receives life insurance amount?"
    ]

    for q in smart_queries:

        smart_search(q)

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    with open("lab10_results_pilot.json", "w") as f:

        json.dump(records, f, indent=2)

    print("\nSaved results to lab10_results.json")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()