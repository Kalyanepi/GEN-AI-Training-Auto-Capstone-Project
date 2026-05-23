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
_CITATION_REQUIRED_TRIGGERS = [
    "deductible", "exclusion", "limit", "total loss", "threshold",
    "section", "chapter", "article",
]


def _fallback_message(reason: str) -> str:
    if reason == LEGAL_ADVICE:
        return (
            "I can't provide legal advice or recommend lawsuits. "
            "For legal questions, please consult an attorney."
        )
    if reason == FAULT_DETERMINATION:
        return (
            "Fault determination is made by your assigned adjuster after "
            "reviewing the police report, photos, and statements. I can help "
            "with coverage details, repair estimates, and claim filing steps."
        )
    if reason == MISSING_CITATION:
        return (
            "I don't have specific policy language on that in my documents. "
            "Please try rephrasing your question or ask about a specific coverage topic."
        )
    if reason == FABRICATED_DATA:
        return (
            "I can only quote repair costs and thresholds from verified data. "
            "Please tell me the damage type and your vehicle category so I "
            "can look up the correct figures."
        )
    return (
        "I can only help with auto insurance topics. "
        "Please try rephrasing your question."
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
    """If the question warrants citations and none provided, block."""
    lower = answer.lower()
    is_fallback = any(phrase in lower for phrase in [
        "don't have specific policy",
        "don't have specific information",
        "contact your adjuster",
        "please contact your adjuster",
    ])
    if is_fallback:
        return None
    if any(trigger in lower for trigger in _CITATION_REQUIRED_TRIGGERS):
        if not citations:
            logger.warning("output_guardrail_missing_citation")
            return OutputDecision(blocked=True, reason=MISSING_CITATION, message=_fallback_message(MISSING_CITATION))
    return None


def _check_fabricated_costs(
    answer: str,
    citations: List[Citation],
    allowed_dollar_values: List[float],
) -> Optional[OutputDecision]:
    """Cross-check every dollar figure in the answer against tool outputs.

    WHY citation exemption: when policy_rag_tool retrieves chunks that contain
    dollar figures (e.g. "$50,000 BI liability limit"), the LLM legitimately
    quotes those numbers in the answer. allowed_dollar_values is only populated
    by CSV tools (repair_cost_tool, total_loss_tool, rental_lookup_tool), so
    RAG-grounded answers will always have an empty allowed list. Blocking them
    as "fabricated" is a false positive — if citations are present, the dollar
    amounts are grounded and legitimate.

    We only block when BOTH conditions are true:
      1. No CSV tool provided allowed_dollar_values (i.e. structured lookup didn't run)
      2. No citations exist to ground the answer (pure LLM fabrication)
    """
    if not allowed_dollar_values:
        # WHY: if citations present, dollar amounts came from retrieved policy
        # chunks — that's legitimate grounding. Don't block.
        if citations:
            return None
        # No citations AND no CSV data = genuinely ungrounded dollar claims.
        suspect = [float(m.group(1).replace(",", "")) for m in _DOLLAR_RE.finditer(answer)]
        suspect_nonzero = [v for v in suspect if v > 0]
        if suspect_nonzero:
            logger.warning("output_guardrail_fabricated_no_source", values=suspect_nonzero)
            return OutputDecision(blocked=True, reason=FABRICATED_DATA, message=_fallback_message(FABRICATED_DATA))
        return None

    # WHY: when citations are also present alongside CSV data, the LLM may
    # legitimately quote dollar amounts from the retrieved PDF chunks (e.g.
    # liability limits, coverage maximums). These won't be in allowed_dollar_values
    # which only contains CSV-tool outputs. Blocking them is a false positive.
    if citations:
        return None

    # WHY: also allow midpoints/averages between any two adjacent allowed values
    # (e.g. low=$200, high=$400 -> midpoint=$300 is legitimate to quote).
    expanded = list(allowed_dollar_values)
    sorted_allowed = sorted(allowed_dollar_values)
    for i in range(len(sorted_allowed) - 1):
        expanded.append((sorted_allowed[i] + sorted_allowed[i + 1]) / 2.0)

    # Use a wider tolerance (20%) since LLMs round and paraphrase numbers.
    tolerance = max(settings.fabricated_cost_tolerance_pct / 100.0, 0.20)
    for match in _DOLLAR_RE.finditer(answer):
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if value <= 0:
            continue
        # WHY: skip small values that are likely percentages or deductible
        # fractions quoted without a dollar context (e.g. "80" in "80% of ACV").
        if value <= 100:
            continue
        if not any(abs(value - allowed) <= max(tolerance * allowed, 10.0) for allowed in expanded):
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
    _citations = citations or []
    _allowed = allowed_dollar_values or []

    for check in (
        lambda: _check_legal_advice(answer),
        lambda: _check_fault(answer),
        lambda: _check_citation_present(answer, _citations),
        lambda: _check_fabricated_costs(answer, _citations, _allowed),
    ):
        decision = check()
        if decision and decision.blocked:
            return decision
    return OutputDecision(blocked=False)
