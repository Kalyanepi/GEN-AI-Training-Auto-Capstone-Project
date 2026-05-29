"""rental_lookup_tool (FR-06) — tier-aware rental reimbursement.

WHY hardcoded tier table backed by RAG: the canonical numbers come from the
Declaration Pages and are critical to never get wrong (architecture plan §2.1).
We surface them deterministically AND attach the supporting RAG citation so
users can verify in the source PDF.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from agent.tools.base_tool import BaseTool, DataNotFoundError, ToolResult
from rag.citation_tracker import chunks_to_citations
from rag.retriever import Retriever


_RENTAL_BY_TIER: Dict[str, Dict[str, Any]] = {
    "standard": {
        "daily_limit_usd": 30,
        "max_days": 15,
        "max_total_usd": 30 * 15,
        "tier_label": "Standard Shield",
    },
    "premium": {
        "daily_limit_usd": 50,
        "max_days": 30,
        "max_total_usd": 50 * 30,
        "tier_label": "Premium Guard",
    },
    "elite": {
        "daily_limit_usd": 75,
        "max_days": 45,
        "max_total_usd": 75 * 45,
        "tier_label": "Comprehensive Elite",
    },
}


class RentalLookupTool(BaseTool):
    name = "rental_lookup_tool"
    description = (
        "Look up rental reimbursement daily limit, max days, and total cap "
        "for a given policy tier. Returns Declaration Page citation."
    )

    def __init__(self, retriever: Optional[Retriever] = None) -> None:
        super().__init__()
        self.retriever = retriever or Retriever()

    async def _execute(self, policy_tier: Optional[str] = None, **_: Any) -> ToolResult:
        if not policy_tier:
            raise DataNotFoundError("policy_tier is required for rental lookup")
        tier = policy_tier.lower()
        if tier not in _RENTAL_BY_TIER:
            raise DataNotFoundError(f"Unknown policy_tier '{policy_tier}'")

        info = _RENTAL_BY_TIER[tier]
        rental_query = f"rental reimbursement daily limit max days {info['tier_label']}"
        chunks = self.retriever.retrieve(
            query=rental_query,
            policy_tier=tier,
            doc_type="rental",
            top_k=3,
        )
        # WHY fallback: small/single-chunk PDFs may not produce rental-tagged
        # chunks for every tier; loosen filters before giving up.
        if not chunks:
            chunks = self.retriever.retrieve(query=rental_query, policy_tier=tier, top_k=3)
        if not chunks:
            chunks = self.retriever.retrieve(query=rental_query, top_k=3)
        citations = chunks_to_citations(chunks)

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "policy_tier": tier,
                "tier_label": info["tier_label"],
                "daily_limit_usd": info["daily_limit_usd"],
                "max_days": info["max_days"],
                "max_total_usd": info["max_total_usd"],
            },
            citations=citations,
            dollar_values=[
                float(info["daily_limit_usd"]),
                float(info["max_total_usd"]),
            ],
        )
