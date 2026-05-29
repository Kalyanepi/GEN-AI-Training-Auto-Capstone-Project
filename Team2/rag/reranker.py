"""Cross-encoder reranker — boosts top-k precision after metadata filtering.

WHY a reranker: bi-encoder embeddings are fast but miss subtle relevance
signals. A cross-encoder scores (query, chunk) jointly and reorders the
already-filtered top candidates — yielding noticeably better top-1/top-3
accuracy without rebuilding the index.

WHY optional (settings.reranker_enabled): cross-encoders add ~50-200ms per
query. Disabled in low-latency demos, enabled when accuracy matters.
"""
from __future__ import annotations

from threading import Lock
from typing import List, Optional

from api.config import settings
from observability.logger import get_logger
from rag.retriever import ChunkResult

logger = get_logger(__name__)

_model = None
_lock = Lock()


def _get_model():
    """Lazy-load the cross-encoder once.

    WHY lazy: importing sentence_transformers triggers torch import and adds
    ~3s to startup. Only pay this cost if reranking is actually used.
    """
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(settings.reranker_model)
        logger.info("reranker_loaded", model=settings.reranker_model)
        return _model


def rerank(query: str, chunks: List[ChunkResult], top_k: Optional[int] = None) -> List[ChunkResult]:
    """Score (query, chunk.text) pairs and reorder by cross-encoder score.

    WHY we keep similarity_score intact and don't overwrite it: downstream
    citation card displays the original semantic similarity for explainability.
    The reranker only changes ORDER, not the displayed score.
    """
    if not settings.reranker_enabled or not chunks:
        return chunks[: top_k or settings.retrieval_top_k]
    try:
        model = _get_model()
        pairs = [(query, c.text) for c in chunks]
        scores = model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: float(x[1]), reverse=True)
        # WHY score cutoff: cross-encoder scores below -2.0 indicate the chunk
        # is not meaningfully related to the query. Dropping them prevents
        # off-topic chunks (e.g. rental PDF returned for a GAP question) from
        # appearing in citations and confusing DeepEval contextual metrics.
        _RERANK_CUTOFF = -2.0
        filtered = [c for c, s in ranked if float(s) >= _RERANK_CUTOFF]
        result = (filtered or [c for c, _ in ranked])[: top_k or settings.retrieval_top_k]
        logger.info("reranking_done", input=len(chunks), output=len(result))
        return result
    except Exception as e:
        # WHY fallback: reranking is an optimization, never a hard requirement.
        # If torch model load fails on a constrained host, original ranking
        # still serves users correctly.
        logger.warning("reranker_failed_fallback", error=str(e))
        return chunks[: top_k or settings.retrieval_top_k]
