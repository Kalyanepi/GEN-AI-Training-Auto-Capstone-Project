"""Metadata-filtered semantic retriever.

WHY over-retrieve + post-filter: FAISS doesn't support metadata filtering
natively. We retrieve top_k * overfetch_multiplier candidates, then post-filter
by tier/coverage/doc_type. This guarantees we never serve Elite-tier deductibles
to Standard-tier users, even when their semantic similarity is high.

WHY the 0.65 similarity threshold: below this score, retrieved chunks are
typically off-topic. Returning them produces hallucination-prone responses.
The retriever returns an empty list instead — the LLM synthesis layer
recognizes this and returns the graceful "no data" message.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agent.cache import LRUCache, normalize_text
from api.config import settings
from ingestion.embedder import Embedder
from observability.logger import get_logger
from rag.faiss_store import FaissStore, get_store

logger = get_logger(__name__)

# Cache the FULL retrieval result (filtered + scored chunks) for a given
# (query, filter, top_k, threshold) tuple. Saves the embed → FAISS search →
# post-filter pipeline on repeat queries. Embedding cache is upstream and
# still helps when the query is the same but filters differ.
_retrieval_cache: LRUCache[Tuple, List["ChunkResult"]] = LRUCache(
    name="retrieval_results",
    max_size=settings.retrieval_cache_size,
    ttl_seconds=settings.retrieval_cache_ttl_seconds,
)


@dataclass
class ChunkResult:
    """A retrieved chunk with score, ready for citation + LLM context."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    section_title: Optional[str]
    subsection_title: Optional[str]
    coverage_type: List[str]
    policy_tier: List[str]
    doc_type: str
    similarity_score: float
    metadata: Dict = field(default_factory=dict)


class Retriever:
    """Embeds queries and returns metadata-filtered top-k chunks."""

    def __init__(self, store: Optional[FaissStore] = None, embedder: Optional[Embedder] = None) -> None:
        self.store = store or get_store()
        self.embedder = embedder or Embedder()

    def retrieve(
        self,
        query: str,
        coverage_type: Optional[str] = None,
        policy_tier: Optional[str] = None,
        doc_type: Optional[str] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[ChunkResult]:
        """Retrieve top_k chunks matching metadata filters above similarity threshold."""
        top_k = top_k or settings.retrieval_top_k
        threshold = similarity_threshold if similarity_threshold is not None else settings.similarity_threshold
        overfetch = top_k * settings.retrieval_overfetch_multiplier

        # Cache key: include all knobs that change the result set.
        cache_key = (
            normalize_text(query),
            coverage_type or "",
            policy_tier or "",
            doc_type or "",
            top_k,
            round(threshold, 4),
        )
        cached = _retrieval_cache.get(cache_key)
        if cached is not None:
            logger.info(
                "retrieval_cache_hit",
                query_preview=query[:80],
                returned=len(cached),
            )
            return cached

        # 1. Embed query (already L2-normalized by Embedder; itself cached).
        query_vec = self.embedder.embed_query(query)

        # 2. Search FAISS — over-retrieve so post-filter has candidates.
        scores, indices = self.store.index.search(query_vec, min(overfetch, self.store.size))

        # 3. Hydrate + filter.
        results: List[ChunkResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            score = float(score)
            if score < threshold:
                continue
            meta = self.store.metadata[idx]
            if not self._matches_filters(meta, coverage_type, policy_tier, doc_type):
                continue
            results.append(self._to_result(meta, score))
            if len(results) >= top_k:
                break

        logger.info(
            "retrieval_done",
            query_preview=query[:80],
            coverage_type=coverage_type,
            policy_tier=policy_tier,
            doc_type=doc_type,
            returned=len(results),
            threshold=threshold,
        )
        _retrieval_cache.set(cache_key, results)
        return results

    @staticmethod
    def _matches_filters(
        meta: Dict,
        coverage_type: Optional[str],
        policy_tier: Optional[str],
        doc_type: Optional[str],
    ) -> bool:
        """True iff chunk matches all provided filters.

        WHY 'general' coverage and tier-agnostic chunks pass: glossary entries
        and definitional clauses apply across coverage types and tiers.
        """
        if coverage_type:
            cov = meta.get("coverage_type") or []
            if coverage_type not in cov and "general" not in cov:
                return False
        if policy_tier:
            tiers = meta.get("policy_tier") or []
            if policy_tier not in tiers:
                return False
        if doc_type:
            if meta.get("doc_type") != doc_type:
                return False
        return True

    @staticmethod
    def _to_result(meta: Dict, score: float) -> ChunkResult:
        return ChunkResult(
            chunk_id=meta.get("chunk_id", ""),
            text=meta.get("text", ""),
            source_file=meta.get("source_file", ""),
            page_number=int(meta.get("page_number", 0)),
            section_title=meta.get("section_title"),
            subsection_title=meta.get("subsection_title"),
            coverage_type=list(meta.get("coverage_type", [])),
            policy_tier=list(meta.get("policy_tier", [])),
            doc_type=meta.get("doc_type", "other"),
            similarity_score=score,
            metadata=meta,
        )
