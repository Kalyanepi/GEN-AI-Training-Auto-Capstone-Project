"""Post-synthesis output guardrails.

WHY post-synthesis: even with grounded retrieval, the LLM might phrase an
answer in a way that crosses prohibited lines (legal advice, fault
determination) or fail to include a citation. This is the last line of defense
before the answer reaches the user.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from api.config import settings
from observability.logger import get_logger
from rag.citation_tracker import Citation

logger = get_logger(__name__)


# Block codes (mirror architecture plan §11.2).
LEGAL_ADVICE = "LEGAL_ADVICE"
FAULT_DETERMINATION = "FAULT_DETERMINATION"
MISSING_CITATION = "MISSING_CITATION"
FABRICATED_DATA = "FABRICATED_DATA"


@dataclass
class OutputDecision:
    blocked: bool
    reason: Optional[str] = None
    message: Optional[str] = None


_LEGAL_ADVICE_PATTERNS = [
    re.compile(r"\byou\s+(should|must|need\s+to)\s+(sue|file\s+a\s+lawsuit|hire\s+a\s+lawyer)\b", re.IGNORECASE),
    re.compile(r"\blegally\s+(liable|responsible|obligated)\b", re.IGNORECASE),
    re.compile(r"\b(get\s+a\s+lawyer|consult\s+(?:an?\s+)?attorney)\b", re.IGNORECASE),
]

_FAULT_PATTERNS = [
    re.compile(r"\b(?:the\s+)?other\s+driver\s+(?:was|is)\s+(?:at\s+fault|responsible)\b", re.IGNORECASE),
    re.compile(r"\byou\s+(?:were|are)\s+(?:not\s+)?at\s+fault\b", re.IGNORECASE),
    re.compile(r"\bclearly\s+(?:not\s+)?your\s+fault\b", re.IGNORECASE),
]

_DOLLAR_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)")

# Coverage-question keywords that REQUIRE a citation in the final answer.
# WHY narrow list: "coverage", "covered", "policy" match virtually every question
# and force citations even when the tool succeeded with deterministic data
# (e.g., tier benefits from CSV). We only block when the answer makes a
# specific factual claim (deductible amount, exclusion, limit, threshold,
# total loss ratio, or references a specific section number) — those need
# to trace back to a source.
_CITATION_REQUIRED_TRIGGERS = [
    "deductible", "exclusion", "limit", "total loss", "threshold",
    "section", "chapter", "article",
]


def _fallback_message(reason: str) -> str:
    if reason == LEGAL_ADVICE:
        return (
            "I can't provide legal advice or recommend lawsuits. For legal "
            f"questions, please consult an attorney. For claim guidance, "
            f"contact your adjuster: {settings.adjuster_phone}."
        )
    if reason == FAULT_DETERMINATION:
        return (
            "Fault determination is made by your assigned adjuster after "
            "reviewing the police report, photos, and statements. I can help "
            "with coverage details, repair estimates, and claim filing steps. "
            f"Adjuster: {settings.adjuster_phone}."
        )
    if reason == MISSING_CITATION:
        return (
            "I don't have specific policy language on that in my documents. "
            f"Please contact your RoadGuard adjuster at {settings.adjuster_phone} "
            "for an official answer."
        )
    if reason == FABRICATED_DATA:
        return (
            "I can only quote repair costs and thresholds from verified data. "
            "Please tell me the damage type and your vehicle category so I "
            "can look up the correct figures."
        )
    return (
        f"Please contact your RoadGuard adjuster at {settings.adjuster_phone} "
        "for assistance."
    )


def _check_legal_advice(answer: str) -> Optional[OutputDecision]:
    for p in _LEGAL_ADVICE_PATTERNS:
        if p.search(answer):
            logger.warning("output_guardrail_legal_advice", pattern=p.pattern)
            return OutputDecision(blocked=True, reason=LEGAL_ADVICE, message=_fallback_message(LEGAL_ADVICE))
    return None


def _check_fault(answer: str) -> Optional[OutputDecision]:
    for p in _FAULT_PATTERNS:
        if p.search(answer):
            logger.warning("output_guardrail_fault", pattern=p.pattern)
            return OutputDecision(blocked=True, reason=FAULT_DETERMINATION, message=_fallback_message(FAULT_DETERMINATION))
    return None


def _check_citation_present(answer: str, citations: List[Citation]) -> Optional[OutputDecision]:
    """If the question warrants citations and none provided, block.

    WHY trigger-list approach: forcing citations on every greeting ("Hi!")
    would produce ugly UX. Only enforce when the answer text contains
    coverage/cost terms.

    WHY fallback exemption: if the answer already says "I don't have specific
    policy language" or "contact your adjuster", it's a legitimate no-data
    response. Blocking it with MISSING_CITATION replaces a graceful answer with
    the same text — pointless and confusing.
    """
    lower = answer.lower()
    is_fallback = any(phrase in lower for phrase in [
        "don't have specific policy",
        "don't have specific information",
        "contact your adjuster",
        "please contact your roadguard",
    ])
    if is_fallback:
        return None
    if any(trigger in lower for trigger in _CITATION_REQUIRED_TRIGGERS):
        if not citations:
            logger.warning("output_guardrail_missing_citation")
            return OutputDecision(blocked=True, reason=MISSING_CITATION, message=_fallback_message(MISSING_CITATION))
    return None


def _check_fabricated_costs(answer: str, allowed_dollar_values: List[float]) -> Optional[OutputDecision]:
    """Cross-check every dollar figure in the answer against tool outputs.

    WHY 10% tolerance: synthesized answers may round ($4,200 -> ~$4,200) or
    quote ranges. Within 10% of a tool-provided value is acceptable; outside
    that, the LLM is fabricating.
    """
    if not allowed_dollar_values:
        # WHY: no tool returned monetary data, so any $ figure in the answer
        # is suspect. But avoid false positives on $0 / generic phrasing —
        # only block if the answer claims specific non-zero amounts.
        suspect = [float(m.group(1).replace(",", "")) for m in _DOLLAR_RE.finditer(answer)]
        suspect_nonzero = [v for v in suspect if v > 0]
        if suspect_nonzero:
            logger.warning("output_guardrail_fabricated_no_source", values=suspect_nonzero)
            return OutputDecision(blocked=True, reason=FABRICATED_DATA, message=_fallback_message(FABRICATED_DATA))
        return None

    tolerance = settings.fabricated_cost_tolerance_pct / 100.0
    for match in _DOLLAR_RE.finditer(answer):
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if value <= 0:
            continue
        # WHY: any dollar figure within tolerance of ANY allowed value is OK.
        # This handles ranges (low/high/avg all listed) and computed values
        # like settlement = ACV - deductible.
        if not any(abs(value - allowed) <= max(tolerance * allowed, 1.0) for allowed in allowed_dollar_values):
            logger.warning(
                "output_guardrail_fabricated_value",
                value=value,
                allowed=allowed_dollar_values,
            )
            return OutputDecision(blocked=True, reason=FABRICATED_DATA, message=_fallback_message(FABRICATED_DATA))
    return None


def check_output(
    answer: str,
    citations: List[Citation],
    allowed_dollar_values: Optional[List[float]] = None,
) -> OutputDecision:
    """Run all output guardrails. First blocker wins."""
    for check in (
        lambda: _check_legal_advice(answer),
        lambda: _check_fault(answer),
        lambda: _check_citation_present(answer, citations),
        lambda: _check_fabricated_costs(answer, allowed_dollar_values or []),
    ):
        decision = check()
        if decision and decision.blocked:
            return decision
    return OutputDecision(blocked=False)
