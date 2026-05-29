"""um_uim_tool — RAG over UM/UIM guide + tier limit injection.

WHY hybrid: tier-specific BI/UM/UIM dollar limits live on the Declaration
Pages while the explanatory text (stacking rules, hit-and-run, deadlines)
lives in the UM_UIM Coverage Guide. We surface BOTH so the LLM has all the
facts to synthesize a complete tier-correct answer.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from agent.tools.base_tool import BaseTool, DataNotFoundError, ToolResult
from rag.citation_tracker import chunks_to_citations
from rag.retriever import Retriever


# UM/UIM limits per architecture plan §2.1 (mirrors BI Liability for these tiers).
_UM_LIMITS_BY_TIER: Dict[str, Dict[str, Any]] = {
    "standard": {
        "tier_label": "Standard Shield",
        "um_uim_bi_per_person_usd": 50_000,
        "um_uim_bi_per_accident_usd": 100_000,
        "umpd_included": False,
    },
    "premium": {
        "tier_label": "Premium Guard",
        "um_uim_bi_per_person_usd": 100_000,
        "um_uim_bi_per_accident_usd": 300_000,
        "umpd_included": False,
    },
    "elite": {
        "tier_label": "Comprehensive Elite",
        "um_uim_bi_per_person_usd": 250_000,
        "um_uim_bi_per_accident_usd": 500_000,
        "umpd_included": True,
    },
}


class UmUimTool(BaseTool):
    name = "um_uim_tool"
    description = (
        "Answer Uninsured / Underinsured Motorist questions — RAG over UM/UIM "
        "Coverage Guide and inject the policyholder's tier-specific limits."
    )

    def __init__(self, retriever: Optional[Retriever] = None) -> None:
        super().__init__()
        self.retriever = retriever or Retriever()

    async def _execute(
        self,
        query: str,
        policy_tier: Optional[str] = None,
        **_: Any,
    ) -> ToolResult:
        if not query:
            raise DataNotFoundError("query is required")

        tier = policy_tier.lower() if policy_tier else None
        chunks = self.retriever.retrieve(query=query, policy_tier=tier, doc_type="um_uim", top_k=5)
        # WHY fallback: drop doc_type filter then tier filter to find anything relevant.
        if not chunks:
            chunks = self.retriever.retrieve(query=query, policy_tier=tier, top_k=5)
        if not chunks:
            chunks = self.retriever.retrieve(query=query, top_k=5)
        citations = chunks_to_citations(chunks)

        tier_limits = None
        dollars: list[float] = []
        if policy_tier:
            tier = policy_tier.lower()
            tier_limits = _UM_LIMITS_BY_TIER.get(tier)
            if tier_limits:
                dollars = [
                    float(tier_limits["um_uim_bi_per_person_usd"]),
                    float(tier_limits["um_uim_bi_per_accident_usd"]),
                ]

        return ToolResult(
            tool_name=self.name,
            success=bool(chunks) or tier_limits is not None,
            data={
                "query": query,
                "policy_tier": policy_tier,
                "tier_limits": tier_limits,
                "chunks": [
                    {
                        "text": c.text,
                        "source_file": c.source_file,
                        "page_number": c.page_number,
                        "section_title": c.section_title,
                        "similarity_score": c.similarity_score,
                    }
                    for c in chunks
                ],
            },
            citations=citations,
            dollar_values=dollars,
        )
