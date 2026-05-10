# lab10/semantic_filter_search.py

import os
import sys
import time
import json

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Shared corpus
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