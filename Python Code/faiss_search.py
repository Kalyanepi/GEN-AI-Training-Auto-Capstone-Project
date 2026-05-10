# lab9/faiss_search.py

import os
import sys
import time
import json

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load shared corpus
sys.path.insert(0, "//home/ubuntu/insurance_lab/shared")
from corpus1 import CORPUS1

# Load environment variables
load_dotenv(dotenv_path="/home/ubuntu/insurance_lab/.env")

# OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

print(f"Corpus loaded: {len(CORPUS1)} documents")


# =========================================================
# BUILD FAISS INDEX
# =========================================================

def build_index(docs, save_path="faiss_insurance_index"):

    print("Building FAISS index (embedding API calls)...")

    t0 = time.time()

    vs = FAISS.from_documents(docs, embeddings)

    elapsed = round(time.time() - t0, 2)

    vs.save_local(save_path)

    print(f"Done in {elapsed}s -> saved to ./{save_path}/")

    return vs, elapsed


# =========================================================
# LOAD EXISTING INDEX
# =========================================================

def load_index(save_path="faiss_insurance_index"):

    vs = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print(f"Index loaded from disk: {save_path}")

    return vs


# =========================================================
# SEARCH FUNCTION
# =========================================================

def search(vs, query, k=3):

    t0 = time.time()

    hits = vs.similarity_search_with_score(query, k=k)

    latency = round((time.time() - t0) * 1000, 1)

    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print(f"Latency: {latency} ms")
    print(f"Top-{k} Results")
    print("=" * 70)

    for rank, (doc, score) in enumerate(hits, start=1):

        m = doc.metadata

        print(
            f"{rank}. "
            f"[{m['doc_id']}] "
            f"{m['policy_type']}/{m['claim_type']} "
            f"score={score:.4f}"
        )

        print(f"   {doc.page_content[:110]}...")
        print()

    return hits, latency


# =========================================================
# MAIN
# =========================================================

def main():

    # Build index
    vs, build_time = build_index(CORPUS1)

    # Test queries
    queries = [

        "How do I file a cashless hospital claim?",

        "What is the no claim bonus for motor insurance?",

        "Is my car engine covered if flooded during monsoon?",

        "What documents are needed to claim life insurance after death?",

        "who is modi",

        # Challenge query
        "What is the free look cancellation period for life insurance?"
    ]

    results = {}

    for q in queries:

        hits, latency = search(vs, q, k=3)

        results[q] = {
            "top_doc": hits[0][0].metadata["doc_id"],
            "score": round(float(hits[0][1]), 4),
            "latency_ms": latency
        }

    # Save results
    with open("lab9_results1.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to lab9_results.json")


if __name__ == "__main__":
    main()