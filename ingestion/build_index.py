"""Entry point: load PDFs -> chunk -> tag -> validate -> embed -> persist FAISS.

Usage:
    python -m ingestion.build_index

WHY a single entry point: ingestions are run rarely (only when source PDFs
change). One reproducible script avoids "which script did we run last time?"
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from api.config import settings
from ingestion.chunk_validator import validate_chunks
from ingestion.embedder import Embedder
from ingestion.metadata_tagger import tag_chunks
from ingestion.pdf_loader import chunk_all_pdfs
from observability.logger import get_logger

logger = get_logger(__name__)


def _persist_index(
    index: faiss.Index,
    metadata: List[Dict],
    out_dir: Path,
) -> None:
    """Persist FAISS index + metadata sidecar to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "index.pkl", "wb") as f:
        pickle.dump(metadata, f)
    # Also write JSON for human inspection / debugging.
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        # WHY: drop the raw text from JSON sidecar to keep it small; the .pkl
        # remains the source of truth for the application.
        compact = [{k: v for k, v in m.items() if k != "text"} for m in metadata]
        json.dump(compact, f, indent=2, default=str)
    logger.info("index_persisted", out_dir=str(out_dir), chunks=len(metadata))


def build() -> None:
    """Execute the full ingestion pipeline."""
    logger.info("ingestion_start", pdf_dir=str(settings.pdf_dir))

    # 1. Chunk all PDFs (Pass 1 + Pass 2).
    raw_chunks = list(chunk_all_pdfs(settings.pdf_dir))
    if not raw_chunks:
        raise RuntimeError(f"No chunks extracted from PDFs in {settings.pdf_dir}")

    # 2. Tag with metadata (coverage, tier, doc_type, flags).
    tagged = tag_chunks(raw_chunks)

    # 3. Validate (drop empty/short/noise/incomplete).
    validated = validate_chunks(tagged)
    if not validated:
        raise RuntimeError("All chunks dropped by validator — check source quality.")

    # 4. Embed (batched + retry + L2 normalize).
    embedder = Embedder()
    texts = [c["text"] for c in validated]
    vectors = embedder.embed_documents(texts)
    logger.info("embeddings_complete", count=len(vectors), dim=vectors.shape[1])

    # 5. Build FAISS IndexFlatIP — inner product on normalized vectors == cosine.
    # WHY Flat over IVF/HNSW: at ~260 chunks, flat is instant, exact, and
    # introduces zero approximation error.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # 6. Persist to disk.
    _persist_index(index, validated, settings.faiss_index_dir)

    logger.info(
        "ingestion_complete",
        total_chunks=len(validated),
        index_path=str(settings.faiss_index_dir),
    )


if __name__ == "__main__":
    build()
