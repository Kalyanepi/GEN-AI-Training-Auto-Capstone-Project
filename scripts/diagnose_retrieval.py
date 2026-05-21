"""Diagnostic script — tests FAISS retrieval directly without the API layer.

Usage:
    python scripts/diagnose_retrieval.py "What is New Car Replacement"
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from ingestion.embedder import Embedder
from rag.faiss_store import get_store


def main() -> None:
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is New Car Replacement"
    print(f"Query: {query}")
    print()

    store = get_store()
    embedder = Embedder()

    print(f"Index size: {store.size}")
    print()

    query_vec = embedder.embed_query(query)
    scores, indices = store.index.search(query_vec, min(20, store.size))

    print("Raw FAISS results (top 20, unfiltered):")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx == -1:
            print(f"  {i}. idx=-1 (no result)")
            continue
        meta = store.metadata[idx]
        print(
            f"  {i}. score={score:.4f}  idx={idx}  "
            f"source={meta.get('source_file','?')}  "
            f"doc_type={meta.get('doc_type','?')}  "
            f"coverage={meta.get('coverage_type',[])}  "
            f"tiers={meta.get('policy_tier',[])}"
        )
        text_preview = meta.get('text', '')[:120].replace('\n', ' ')
        print(f"      text: {text_preview}...")
        print()

    # Also test with the actual retriever to see filtering effects.
    from rag.retriever import Retriever
    retriever = Retriever(store=store, embedder=embedder)

    print("Retriever with NO filters:")
    results = retriever.retrieve(query, top_k=10)
    print(f"  returned={len(results)}")
    for r in results:
        print(
            f"    score={r.similarity_score:.4f}  "
            f"doc_type={r.doc_type}  source={r.source_file}  "
            f"coverage={r.coverage_type}  tiers={r.policy_tier}"
        )
    print()

    print("Retriever with doc_type='fnol':")
    results_fnol = retriever.retrieve(query, doc_type="fnol", top_k=10)
    print(f"  returned={len(results_fnol)}")
    for r in results_fnol:
        print(f"    score={r.similarity_score:.4f}  source={r.source_file}")


if __name__ == "__main__":
    main()
