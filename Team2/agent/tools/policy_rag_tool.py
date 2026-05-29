"""policy_rag_tool — FR-01, FR-02, FR-09: metadata-filtered RAG over policy PDFs."""
from __future__ import annotations

from typing import Any, List, Optional

from agent.tools.base_tool import BaseTool, ToolResult
from rag.citation_tracker import chunks_to_citations
from rag.reranker import rerank
from rag.retriever import Retriever


class PolicyRagTool(BaseTool):
    name = "policy_rag_tool"
    description = (
        "Retrieve policy text grounded answers for coverage questions, "
        "exclusions, and definitions. Filters by coverage_type and policy_tier."
    )

    def __init__(self, retriever: Optional[Retriever] = None) -> None:
        super().__init__()
        self.retriever = retriever or Retriever()

    async def _execute(
        self,
        query: str,
        coverage_type: Optional[str] = None,
        policy_tier: Optional[str] = None,
        doc_type: Optional[str] = None,
        top_k: Optional[int] = None,
        **_: Any,
    ) -> ToolResult:
        # WHY progressive retrieval: with a small index, strict filters
        # (tier + coverage + doc_type) can return nothing even when relevant
        # chunks exist under looser tagging. We progressively relax until
        # we find candidates, then rerank.
        def _try_retrieve(**kwargs) -> list:
            candidates = self.retriever.retrieve(
                query=query, top_k=(top_k or 0) * 2 if top_k else None, **kwargs
            )
            return candidates

        # Stage 1: full filter
        candidates = _try_retrieve(
            coverage_type=coverage_type, policy_tier=policy_tier, doc_type=doc_type
        )
        # Stage 2: drop doc_type (e.g. fnol chunks may not match "policy" doc_type)
        if not candidates and doc_type:
            candidates = _try_retrieve(coverage_type=coverage_type, policy_tier=policy_tier)
        # Stage 3: drop coverage_type (e.g. "New Car Replacement" not in coverage keywords)
        if not candidates and coverage_type:
            candidates = _try_retrieve(policy_tier=policy_tier)
        # Stage 4: drop tier filter (cross-tier questions like "what tiers include X")
        if not candidates and policy_tier:
            candidates = _try_retrieve()
        # Stage 5: last resort — any chunk
        if not candidates:
            candidates = _try_retrieve()

        ranked = rerank(query, candidates, top_k=top_k)
        citations = chunks_to_citations(ranked)
        return ToolResult(
            tool_name=self.name,
            success=bool(ranked),
            data={
                "query": query,
                "coverage_type": coverage_type,
                "policy_tier": policy_tier,
                "doc_type": doc_type,
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "source_file": c.source_file,
                        "page_number": c.page_number,
                        "section_title": c.section_title,
                        "similarity_score": c.similarity_score,
                    }
                    for c in ranked
                ],
            },
            citations=citations,
            fallback_message=self._no_data_message() if not ranked else None,
        )

    def _no_data_message(self) -> str:
        return (
            "I don't have specific policy language on that in my documents. "
            "Please contact your RoadGuard adjuster for an official answer."
        )
