"""
RAG Retrieval Improvements
==========================

This file demonstrates three retrieval improvements in LangChain:

1. MMR Retrieval
   - Prevents top-k saturation
   - Improves diversity of retrieved chunks

2. Similarity Score Threshold
   - Rejects irrelevant queries
   - Ensures minimum semantic similarity

3. Metadata Filtering
   - Restricts retrieval to specific domains
   - Useful for multi-domain knowledge bases
"""

from langchain.schema import Document


# ============================================================
# FIX 1 — MMR Retrieval to Avoid Top-K Saturation
# ============================================================

# Diversified retrieval using Maximum Marginal Relevance (MMR)

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,              # Final chunks to return
        "fetch_k": 20,       # Candidate chunks to consider
        "lambda_mult": 0.7,  # 0 = max diversity, 1 = max relevance
    },
)

print("MMR retriever configured successfully.")


# ============================================================
# FIX 2 — Similarity Score Threshold
# ============================================================

# Reject retrieval if similarity score is too low

threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 4,
        "score_threshold": 0.75,  # Minimum similarity threshold
    },
)

print("Similarity threshold retriever configured successfully.")


# Example query execution
query = "What is the company leave policy?"

result = chain.invoke({"query": query})

# Check whether any relevant documents were found
if not result["source_documents"]:
    print("No relevant documents found for this query.")
else:
    print("Relevant documents retrieved successfully.")


# ============================================================
# FIX 3 — Metadata Filtering for Domain-Scoped Retrieval
# ============================================================

# Example: Tag chunks with metadata during ingestion

for chunk in chunks:

    if "policy.txt" in chunk.metadata["source"]:
        chunk.metadata["domain"] = "hr_policy"

    elif "handbook.txt" in chunk.metadata["source"]:
        chunk.metadata["domain"] = "it_security"


# Create retriever restricted to HR policy documents only

filtered_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {
            "domain": "hr_policy"
        },
    },
)

print("Metadata-filtered retriever configured successfully.")