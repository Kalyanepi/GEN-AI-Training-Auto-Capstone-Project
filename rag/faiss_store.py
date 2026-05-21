"""FAISS index loader — singleton accessor.

WHY singleton: loading the index + metadata takes ~100ms; doing it on every
query would add unacceptable latency. Loaded once at API startup, reused.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import faiss

from api.config import settings
from observability.logger import get_logger

logger = get_logger(__name__)


class FaissStore:
    """Holds the FAISS index and parallel metadata list (index aligned)."""

    def __init__(self, index: faiss.Index, metadata: List[Dict]) -> None:
        self.index = index
        self.metadata = metadata

    @property
    def size(self) -> int:
        return len(self.metadata)

    @property
    def dim(self) -> int:
        return self.index.d


_store: Optional[FaissStore] = None
_lock = Lock()


def load_store(index_dir: Path | None = None) -> FaissStore:
    """Load FAISS index + metadata from disk (idempotent)."""
    global _store
    with _lock:
        if _store is not None:
            return _store
        index_dir = index_dir or settings.faiss_index_dir
        index_path = index_dir / "index.faiss"
        meta_path = index_dir / "index.pkl"
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_dir}. "
                f"Run: python -m ingestion.build_index"
            )
        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        if index.ntotal != len(metadata):
            raise RuntimeError(
                f"Index/metadata size mismatch: {index.ntotal} vs {len(metadata)}"
            )
        _store = FaissStore(index=index, metadata=metadata)
        logger.info("faiss_store_loaded", chunks=_store.size, dim=_store.dim)
        return _store


def get_store() -> FaissStore:
    """Get the loaded singleton; load lazily if needed."""
    if _store is None:
        return load_store()
    return _store


def reset_store() -> None:
    """Clear the singleton — used by tests."""
    global _store
    with _lock:
        _store = None
