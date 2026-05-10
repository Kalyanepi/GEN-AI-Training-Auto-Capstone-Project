
# lab11/hybrid_search_streamlit.py

import os
import sys
import time
import json
import hashlib
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="InsureSafe AI Assistant",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ InsureSafe AI Assistant")
st.caption("Ask me anything about insurance policies, claims, and coverage")


# =========================================================
# LOAD CORPUS (shared)
# =========================================================
# Keep your shared path logic
sys.path.insert(0, "/home/ubuntu/insurance_lab/shared")
from corpus import CORPUS  # noqa: E402


# =========================================================
# ENV
# =========================================================
load_dotenv(dotenv_path="/home/ubuntu/insurance_lab/.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not found. Set it in /home/ubuntu/insurance_lab/.env")


# =========================================================
# HELPERS
# =========================================================
def safe_doc_id(doc) -> str:
    """
    Ensure every doc has a stable doc_id for fusion & display.
    Uses metadata.doc_id if present; otherwise hashes content.
    """
    md = getattr(doc, "metadata", {}) or {}
    if "doc_id" in md and md["doc_id"]:
        return str(md["doc_id"])

    content = getattr(doc, "page_content", "") or ""
    return "HASH_" + hashlib.md5(content.encode("utf-8")).hexdigest()[:10]


def infer_policy_type(query: str):
    q = query.lower()

    if any(word in q for word in ["car", "motor", "vehicle", "garage", "engine", "idv", "accident", "flood"]):
        return "motor"

    if any(word in q for word in ["hospital", "health", "cashless", "surgery", "maternity", "diabetes"]):
        return "health"

    if any(word in q for word in ["life", "death", "nominee", "policyholder", "family", "pass away"]):
        return "life"

    return None


# =========================================================
# CACHED RESOURCES (IMPORTANT FOR STREAMLIT)
# =========================================================
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )


@st.cache_resource(show_spinner=False)
def build_bm25_index():
    texts = [doc.page_content for doc in CORPUS]
    tokenized = [t.lower().split() for t in texts]
    return BM25Okapi(tokenized)


@st.cache_resource(show_spinner=False)
def load_or_build_faiss():
    embeddings = get_embeddings()
    idx_path = "/home/ubuntu/insurance_lab/lab9/faiss_insurance_index"

    if os.path.exists(idx_path):
        return FAISS.load_local(
            idx_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    faiss_vs = FAISS.from_documents(CORPUS, embeddings)
    faiss_vs.save_local(idx_path)
    return faiss_vs


# Build / load indexes once
bm25 = build_bm25_index()
faiss_vs = load_or_build_faiss()


# =========================================================
# SEARCH ENGINES
# =========================================================
def bm25_search(query: str, k: int = 6):
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    for i in top_idx:
        results.append((CORPUS[i], float(scores[i])))
    return results


def faiss_search(query: str, k: int = 6, use_filter: bool = False):
    if use_filter:
        policy_type = infer_policy_type(query)
        if policy_type:
            return faiss_vs.similarity_search_with_score(
                query,
                k=k,
                filter={"policy_type": policy_type}
            )
    return faiss_vs.similarity_search_with_score(query, k=k)


def rrf_fuse(list1, list2, k: int = 60):
    """
    Reciprocal Rank Fusion:
      fused_score += 1 / (k + rank)
    """
    scores = {}
    docs = {}

    for rank, (doc, _) in enumerate(list1, start=1):
        doc_id = safe_doc_id(doc)
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs[doc_id] = doc

    for rank, (doc, _) in enumerate(list2, start=1):
        doc_id = safe_doc_id(doc)
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs[doc_id] = doc

    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(docs[d], float(scores[d])) for d in ranked]


def hybrid_search(query: str, k: int = 3, use_filter: bool = False):
    t0 = time.time()

    bm25_results = bm25_search(query, k=k * 2)
    faiss_results = faiss_search(query, k=k * 2, use_filter=use_filter)

    hybrid_results = rrf_fuse(bm25_results, faiss_results)[:k]

    latency = round((time.time() - t0) * 1000, 1)

    return {
        "bm25": bm25_results[:k],
        "faiss": faiss_results[:k],
        "hybrid": hybrid_results,
        "latency_ms": latency,
        "policy_filter": infer_policy_type(query) if use_filter else None
    }


# =========================================================
# OPTIONAL: LLM ANSWER USING RETRIEVED CONTEXT
# =========================================================
def make_context(docs_with_scores, max_chars: int = 3500):
    chunks = []
    total = 0
    for doc, score in docs_with_scores:
        doc_id = safe_doc_id(doc)
        text = (doc.page_content or "").strip()
        block = f"[{doc_id} | score={score:.4f}]\n{text}\n"
        if total + len(block) > max_chars:
            break
        chunks.append(block)
        total += len(block)
    return "\n---\n".join(chunks)


def answer_with_llm(query: str, context: str, model: str, temperature: float):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=OPENAI_API_KEY
    )

    system = SystemMessage(content=(
        "You are InsureSafe AI Assistant. Answer strictly using the provided context. "
        "If context is insufficient, say what is missing and ask a clarifying question."
    ))
    human = HumanMessage(content=f"QUESTION:\n{query}\n\nCONTEXT:\n{context}")

    return llm.invoke([system, human]).content


# =========================================================
# SESSION STATE INIT
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_results" not in st.session_state:
    st.session_state.last_results = None


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("⚙️ Settings")

    model = st.selectbox("Select Model", ["gpt-4o-mini", "gpt-4o"])
    
    st.divider()

    retrieval_mode = st.radio("Retrieval mode", ["hybrid", "bm25", "faiss"], index=0)
     
    st.divider()

    st.write(f"Total Messages: {len(st.session_state.messages)}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.session_state.last_results = None
            st.rerun()

    with col2:
        chat_json = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            "⬇️ Download chat",
            data=chat_json,
            file_name="insuresafe_chat.json",
            mime="application/json"
        )


# =========================================================
# RENDER CHAT HISTORY
# =========================================================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# =========================================================
# CHAT INPUT
# =========================================================
prompt = st.chat_input("Type your question…")

if prompt:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            results = hybrid_search(prompt, k=top_k, use_filter=use_filter)

            # Pick which docs to show/use
            if retrieval_mode == "bm25":
                chosen = results["bm25"]
            elif retrieval_mode == "faiss":
                chosen = results["faiss"]
            else:
                chosen = results["hybrid"]

            st.session_state.last_results = results

        # Display retrieval stats
        meta = f"**Latency:** {results['latency_ms']} ms"
        if use_filter:
            meta += f"  |  **Policy filter:** `{results['policy_filter']}`"
        st.markdown(meta)

        # Show retrieved docs
        with st.expander("🔎 Retrieved passages", expanded=True):
            for rank, (doc, score) in enumerate(chosen, start=1):
                doc_id = safe_doc_id(doc)
                st.markdown(f"**#{rank} — {doc_id}**  (score: `{score:.4f}`)")
                st.write((doc.page_content or "").strip())
                st.divider()

        # Optionally generate an answer grounded in retrieval
        if generate_answer:
            context = make_context(chosen, max_chars=3500)
            if not context.strip():
                final = "I couldn't build context from retrieved passages. Please try a different query."
            else:
                final = answer_with_llm(prompt, context, model=model, temperature=temperature)

            st.markdown(final)
            st.session_state.messages.append({"role": "assistant", "content": final})
        else:
            # If not generating, still respond politely
            final = "Shown the retrieved passages above. Enable **Generate answer** to get a grounded response."
            st.markdown(final)
            st.session_state.messages.append({"role": "assistant", "content": final})
