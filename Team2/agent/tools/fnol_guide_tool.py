"""fnol_guide_tool (FR-05) — RAG over the FNOL PDF + hardcoded deadline table.

WHY hybrid approach: filing deadlines and tier-specific benefits are
critical facts that must NEVER be wrong or omitted (FR-05 acceptance
condition: "filing deadlines per their policy"). Hardcoding them from the
FNOL_Intake_Guidelines PDF guarantees correctness independent of RAG
retrieval quality. RAG still supplies the step-by-step narrative and
required documentation checklist.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.tools.base_tool import BaseTool, ToolResult
from rag.citation_tracker import chunks_to_citations
from rag.reranker import rerank
from rag.retriever import Retriever


# ---------------------------------------------------------------------------
# Hardcoded filing deadlines from FNOL_Intake_Guidelines_RoadGuard.pdf §1.2
# WHY hardcoded: RAG may not surface the deadline section for every query
# phrasing. These are deterministic facts — hardcoding mirrors the approach
# used by rental_lookup_tool for tier benefit tables.
# ---------------------------------------------------------------------------
_FILING_DEADLINES: Dict[str, str] = {
    "standard":    "File FNOL within 24 hours of discovering the loss. "
                   "Failure to report promptly may affect your claim.",
    "hit_and_run": "Must be filed within 72 hours. Police report required within 24 hours of incident.",
    "uninsured":   "Must be filed within 72 hours. Police report required within 24 hours of incident.",
    "theft":       "Police report required within 24 hours of discovery. "
                   "FNOL must be filed immediately after the police report.",
    "glass_only":  "No strict deadline, but coverage ends at policy expiration. File as soon as practical.",
}

# Tier-specific benefits relevant to FNOL process (from Declaration Pages).
# WHY included here: FR-05 acceptance condition says "per their policy" —
# rental authorization and roadside dispatch are part of the post-FNOL flow.
_TIER_FNOL_BENEFITS: Dict[str, Dict[str, Any]] = {
    "standard": {
        "rental_daily": "$30/day",
        "rental_max_days": "15 days",
        "roadside_towing": "up to $100",
        "roadside_calls": "2 calls/term",
        "medpay": "$5,000/person",
    },
    "premium": {
        "rental_daily": "$50/day",
        "rental_max_days": "30 days",
        "roadside_towing": "up to $150",
        "roadside_calls": "4 calls/term",
        "medpay": "$10,000/person",
    },
    "elite": {
        "rental_daily": "$75/day",
        "rental_max_days": "45 days",
        "roadside_towing": "up to $250",
        "roadside_calls": "Unlimited",
        "medpay": "$25,000/person",
    },
}

# Post-FNOL process steps (from FNOL PDF §4.1) — deterministic, not RAG.
_POST_FNOL_STEPS: List[str] = [
    "Step 1 – Claim number assigned within 30 minutes via text/email.",
    "Step 2 – Adjuster contacts you within 1 business day to schedule inspection.",
    "Step 3 – Vehicle inspection scheduled within 2 business days "
              "(RoadGuard center, mobile inspector, or photos-only for minor damage).",
    "Step 4 – Coverage determination within 5 business days of inspection.",
    "Step 5 – Repair estimate issued or total loss ACV offer made within 5 business days of inspection.",
    "Step 6 – Rental authorization issued simultaneously with repair approval or total loss determination.",
]


class FnolGuideTool(BaseTool):
    name = "fnol_guide_tool"
    description = (
        "Provide step-by-step FNOL (First Notice of Loss) guidance grounded in "
        "the RoadGuard FNOL Intake Guidelines, including filing deadlines and "
        "tier-specific post-FNOL benefits."
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
        parts: List[str] = ["FNOL filing steps required documentation deadlines"]
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

        # Resolve filing deadline: match incident_type keyword to deadline table.
        deadline_key = "standard"
        if incident_type:
            it = incident_type.lower()
            if "hit" in it and "run" in it:
                deadline_key = "hit_and_run"
            elif "uninsured" in it or "um" in it:
                deadline_key = "uninsured"
            elif "theft" in it or "stolen" in it:
                deadline_key = "theft"
            elif "glass" in it or "windshield" in it:
                deadline_key = "glass_only"
        filing_deadline = _FILING_DEADLINES[deadline_key]

        # Resolve tier benefits for post-FNOL context.
        tier_key = (policy_tier or "standard").lower()
        tier_benefits = _TIER_FNOL_BENEFITS.get(tier_key, _TIER_FNOL_BENEFITS["standard"])

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "incident_type": incident_type,
                "has_injuries": has_injuries,
                "has_other_party": has_other_party,
                "policy_tier": tier_key,
                "filing_deadline": filing_deadline,
                "post_fnol_steps": _POST_FNOL_STEPS,
                "tier_benefits": tier_benefits,
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
            "documents. Please refer to your policy documents for filing steps."
        )
