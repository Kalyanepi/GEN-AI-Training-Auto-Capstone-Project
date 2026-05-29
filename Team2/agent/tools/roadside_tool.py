"""roadside_tool — tier-aware roadside / towing benefits."""
from __future__ import annotations

from typing import Any, Dict, Optional

from agent.tools.base_tool import BaseTool, DataNotFoundError, ToolResult
from rag.citation_tracker import chunks_to_citations
from rag.retriever import Retriever


# Tier benefit table per architecture plan §2.1.
_ROADSIDE_BY_TIER: Dict[str, Dict[str, Any]] = {
    "standard": {
        "tier_label": "Standard Shield",
        "tow_per_call_usd": 75,
        "tow_calls_per_term": 2,
        "lockout_per_term": 1,
        "winching": "Not included",
        "trip_interruption_usd": 0,
    },
    "premium": {
        "tier_label": "Premium Guard",
        "tow_per_call_usd": 150,
        "tow_calls_per_term": 4,
        "lockout_per_term": 2,
        "winching": "Up to 100 ft",
        "trip_interruption_usd": 300,
    },
    "elite": {
        "tier_label": "Comprehensive Elite",
        "tow_per_call_usd": 250,
        "tow_calls_per_term": "Unlimited",
        "lockout_per_term": "Unlimited",
        "winching": "Up to 300 ft",
        "trip_interruption_usd": 600,
    },
}


class RoadsideTool(BaseTool):
    name = "roadside_tool"
    description = (
        "Look up roadside benefits — towing, lockout, winching, trip "
        "interruption — for the policyholder's tier."
    )

    def __init__(self, retriever: Optional[Retriever] = None) -> None:
        super().__init__()
        self.retriever = retriever or Retriever()

    async def _execute(
        self,
        policy_tier: Optional[str] = None,
        service_type: Optional[str] = None,
        **_: Any,
    ) -> ToolResult:
        if not policy_tier:
            raise DataNotFoundError("policy_tier is required for roadside lookup")
        tier = policy_tier.lower()
        if tier not in _ROADSIDE_BY_TIER:
            raise DataNotFoundError(f"Unknown policy_tier '{policy_tier}'")

        info = _ROADSIDE_BY_TIER[tier]
        query_parts = [info["tier_label"], "roadside"]
        if service_type:
            query_parts.append(service_type)
        composed = " ".join(query_parts)
        chunks = self.retriever.retrieve(query=composed, policy_tier=tier, doc_type="roadside", top_k=3)
        # WHY fallback: loosen filters when no roadside-tagged chunks match.
        if not chunks:
            chunks = self.retriever.retrieve(query=composed, policy_tier=tier, top_k=3)
        if not chunks:
            chunks = self.retriever.retrieve(query=composed, top_k=3)
        citations = chunks_to_citations(chunks)

        # WHY we coerce numeric tier benefits into dollar_values: enables the
        # output guardrail's fabricated-cost check to whitelist these figures.
        dollars = [float(info["tow_per_call_usd"])]
        if isinstance(info["trip_interruption_usd"], (int, float)) and info["trip_interruption_usd"]:
            dollars.append(float(info["trip_interruption_usd"]))

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "policy_tier": tier,
                "service_type": service_type,
                **info,
            },
            citations=citations,
            dollar_values=dollars,
        )
