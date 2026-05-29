"""OpenAI embedding pipeline with batching, retry, and L2 normalization.

WHY L2-normalize before FAISS insert: we use IndexFlatIP (inner product),
which equals cosine similarity ONLY on unit-length vectors. Skipping
normalization silently produces wrong rankings.
"""
from __future__ import annotations

from typing import List

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError, APITimeoutError

from agent.cache import LRUCache, normalize_text
from api.config import settings
from observability.logger import get_logger

logger = get_logger(__name__)

# Query-embedding cache: identical user queries produce identical vectors,
# so caching saves an OpenAI round-trip on the hot retrieval path.
_embed_query_cache: LRUCache[str, np.ndarray] = LRUCache(
    name="query_embeddings",
    max_size=settings.embed_cache_size,
    ttl_seconds=settings.embed_cache_ttl_seconds,
)


class Embedder:
    """Batched embedder with retry/backoff."""

    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.batch_size = settings.embedding_batch_size

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError, APITimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Return an (N, dim) float32 L2-normalized matrix.

        WHY float32: FAISS expects float32; passing float64 silently fails on
        some platforms or doubles memory.
        """
        all_vectors: List[List[float]] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors = self._embed_batch(batch)
            all_vectors.extend(vectors)
            logger.info(
                "embedding_batch_done",
                batch=(i // self.batch_size) + 1,
                total_batches=total_batches,
                size=len(batch),
            )

        arr = np.array(all_vectors, dtype=np.float32)
        # L2-normalize — required for IndexFlatIP -> cosine similarity.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero on empty embeddings
        return arr / norms

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query and L2-normalize.

        Returns shape (1, dim) for FAISS compatibility.
        """
        cache_key = normalize_text(text)
        cached = _embed_query_cache.get(cache_key)
        if cached is not None:
            logger.debug("embed_cache_hit", chars=len(text))
            return cached

        vector = self._embed_batch([text])[0]
        arr = np.array([vector], dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr
        result = arr / norm
        _embed_query_cache.set(cache_key, result)
        return result
