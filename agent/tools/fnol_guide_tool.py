"""fnol_guide_tool (FR-05) — RAG over the FNOL PDF + step formatter."""
from __future__ import annotations

from typing import Any, List, Optional

from agent.tools.base_tool import BaseTool, ToolResult
from rag.citation_tracker import chunks_to_citations
from rag.reranker import rerank
from rag.retriever import Retriever


class FnolGuideTool(BaseTool):
    name = "fnol_guide_tool"
    description = (
        "Provide step-by-step FNOL (First Notice of Loss) guidance grounded in "
        "the RoadGuard FNOL Intake Guidelines."
    )

    def __init__(self, retriever: Optional[Retriever] = None) -> None:
        super().__init__()
        self.retriever = retriever or Retriever()

    async def _execute(
        self,
        incident_type: Optional[str] = None,
        policy_tier: Optional[str] = None,
        has_injuries: Optional[bool] = None,
        has_other_party: Optional[bool] = None,
        query: Optional[str] = None,
        **_: Any,
    ) -> ToolResult:
        # WHY constructed query: FNOL questions vary in phrasing; we synthesize
        # a normalized query from structured params + any free-text query.
        parts: List[str] = ["FNOL filing steps"]
        if incident_type:
            parts.append(incident_type)
        if has_injuries:
            parts.append("with injuries")
        if has_other_party:
            parts.append("third party involved")
        if query:
            parts.append(query)
        composed_query = " ".join(parts)

        # WHY two-stage retrieval: prefer FNOL-tagged chunks, but fall back
        # to any policy chunk that mentions claim filing if FNOL chunks are
        # sparse (e.g., tiny FNOL PDF yielded only 1-2 chunks).
        candidates = self.retriever.retrieve(query=composed_query, doc_type="fnol", top_k=8)
        if not candidates:
            candidates = self.retriever.retrieve(query=composed_query, top_k=8)
        ranked = rerank(composed_query, candidates, top_k=5)
        citations = chunks_to_citations(ranked)

        return ToolResult(
            tool_name=self.name,
            success=bool(ranked),
            data={
                "incident_type": incident_type,
                "has_injuries": has_injuries,
                "has_other_party": has_other_party,
                "policy_tier": policy_tier,
                "query": composed_query,
                "chunks": [
                    {
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
            "I couldn't locate FNOL guidance for that incident type in our "
            "documents. Please call RoadGuard Claims directly to file."
        )
