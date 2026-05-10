# lab11/hybrid_search.py

import os
import sys
import time
import json
import numpy as np

from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# =========================================================
# LOAD SHARED CORPUS
# =========================================================

sys.path.insert(0, "/home/ubuntu/insurance_lab/shared")
from corpus import CORPUS

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv(dotenv_path="/home/ubuntu/insurance_lab/.env")

# =========================================================
# OPENAI EMBEDDINGS
# =========================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# =========================================================
# BUILD BM25 INDEX
# =========================================================

texts = [doc.page_content for doc in CORPUS]

tokenized_texts = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_texts)

print("BM25 index built successfully")

# =========================================================
# LOAD OR BUILD FAISS INDEX
# =========================================================

IDX = "/home/ubuntu/insurance_lab/lab9/faiss_insurance_index"

if os.path.exists(IDX):

    faiss_vs = FAISS.load_local(
        IDX,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("FAISS index loaded from Lab 9")

else:

    print("FAISS index not found. Building new index...")

    faiss_vs = FAISS.from_documents(CORPUS, embeddings)

    faiss_vs.save_local(IDX)

    print("New FAISS index built and saved")


# =========================================================
# BM25 SEARCH
# =========================================================

def bm25_search(query, k=6):

    scores = bm25.get_scores(
        query.lower().split()
    )

    top_indices = np.argsort(scores)[::-1][:k]

    results = []

    for idx in top_indices:

        results.append(
            (
                CORPUS[idx],
                float(scores[idx])
            )
        )

    return results


# =========================================================
# RECIPROCAL RANK FUSION (RRF)
# =========================================================

def rrf_fuse(list1, list2, k=60):

    """
    Reciprocal Rank Fusion

    score += 1 / (k + rank)
    """

    scores = {}
    docs = {}

    # BM25 ranks
    for rank, (doc, _) in enumerate(list1, start=1):

        doc_id = doc.metadata["doc_id"]

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1.0 / (k + rank)
        )

        docs[doc_id] = doc

    # FAISS ranks
    for rank, (doc, _) in enumerate(list2, start=1):

        doc_id = doc.metadata["doc_id"]

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1.0 / (k + rank)
        )

        docs[doc_id] = doc

    # Sort by descending fused score
    ranked_doc_ids = sorted(
        scores.keys(),
        key=lambda x: scores[x],
        reverse=True
    )

    return [
        (docs[doc_id], scores[doc_id])
        for doc_id in ranked_doc_ids
    ]


# =========================================================
# SMART POLICY FILTER
# =========================================================

def infer_policy_type(query):

    q = query.lower()

    if any(word in q for word in [
        "car",
        "motor",
        "vehicle",
        "garage",
        "engine",
        "idv",
        "accident"
    ]):
        return "motor"

    if any(word in q for word in [
        "hospital",
        "health",
        "cashless",
        "surgery",
        "maternity",
        "diabetes"
    ]):
        return "health"

    if any(word in q for word in [
        "life",
        "death",
        "nominee",
        "policyholder",
        "family",
        "pass away"
    ]):
        return "life"

    return None


# =========================================================
# HYBRID SEARCH
# =========================================================

def hybrid_search(query, k=3, use_filter=False):

    t0 = time.time()

    # -----------------------------------------------------
    # BM25 RESULTS
    # -----------------------------------------------------

    bm25_results = bm25_search(query, k=k * 2)

    # -----------------------------------------------------
    # FAISS RESULTS
    # -----------------------------------------------------

    if use_filter:

        policy_type = infer_policy_type(query)

        if policy_type:

            faiss_results = faiss_vs.similarity_search_with_score(
                query,
                k=k * 2,
                filter={"policy_type": policy_type}
            )

        else:

            faiss_results = faiss_vs.similarity_search_with_score(
                query,
                k=k * 2
            )

    else:

        faiss_results = faiss_vs.similarity_search_with_score(
            query,
            k=k * 2
        )

    # -----------------------------------------------------
    # RRF FUSION
    # -----------------------------------------------------

    hybrid_results = rrf_fuse(
        bm25_results,
        faiss_results
    )[:k]

    latency = round(
        (time.time() - t0) * 1000,
        1
    )

    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    print(f"Latency: {latency} ms")

    if use_filter:
        print(f"Policy Filter: {infer_policy_type(query)}")

    print("\nRANK | BM25           | FAISS          | HYBRID")
    print("-" * 65)

    for i in range(k):

        bm = (
            bm25_results[i][0].metadata["doc_id"]
            if i < len(bm25_results)
            else "-"
        )

        fa = (
            faiss_results[i][0].metadata["doc_id"]
            if i < len(faiss_results)
            else "-"
        )

        hy = (
            hybrid_results[i][0].metadata["doc_id"]
            if i < len(hybrid_results)
            else "-"
        )

        print(
            f"{i+1:<4} | "
            f"{bm:<15} | "
            f"{fa:<15} | "
            f"{hy:<15}"
        )

    return {
        "bm25": bm25_results[:k],
        "faiss": faiss_results[:k],
        "hybrid": hybrid_results,
        "latency_ms": latency
    }


# =========================================================
# MAIN
# =========================================================

def main():

    queries = [

        (
            "IDV insured declared value depreciation",
            "M009",
            "BM25 wins — exact insurance acronym"
        ),

        (
            "My car got flooded in the rains am I covered",
            "M006",
            "FAISS wins — conceptual query"
        ),

        (
            "NCB for accident free car policy",
            "M003",
            "Hybrid wins — keyword + semantic mix"
        ),

        (
            "what happens to my family if I pass away",
            "L001",
            "FAISS wins — conversational language"
        )
    ]

    records = []

    # =====================================================
    # NORMAL HYBRID SEARCH
    # =====================================================

    print("\n" + "#" * 100)
    print("STANDARD HYBRID SEARCH")
    print("#" * 100)

    for query, expected, note in queries:

        results = hybrid_search(
            query=query,
            k=3,
            use_filter=False
        )

        print(f"\nExpected Top Doc: {expected}")
        print(f"Observation: {note}")

        records.append({
            "query": query,
            "expected": expected,
            "bm25_top1":
                results["bm25"][0][0].metadata["doc_id"],
            "faiss_top1":
                results["faiss"][0][0].metadata["doc_id"],
            "hybrid_top1":
                results["hybrid"][0][0].metadata["doc_id"],
            "latency_ms":
                results["latency_ms"]
        })

    # =====================================================
    # FILTERED HYBRID SEARCH
    # =====================================================

    print("\n" + "#" * 100)
    print("FILTERED HYBRID SEARCH")
    print("#" * 100)

    filtered_query = "NCB for accident free car policy"

    hybrid_search(
        query=filtered_query,
        k=3,
        use_filter=True
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    with open("lab11_results.json", "w") as f:

        json.dump(records, f, indent=2)

    print("\nResults saved to lab11_results.json")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()