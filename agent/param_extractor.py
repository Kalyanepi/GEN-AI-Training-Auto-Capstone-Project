"""Lightweight regex extractor for structured params embedded in chat text.

WHY: users naturally type "ACV is $8,000 and repair $6,500 in Illinois" rather
than filling structured fields. Without extraction, tools like total_loss_tool
receive acv=None and refuse to compute. This extractor parses the common
patterns so the orchestrator can populate AgentState before tool dispatch.

WHY regex (not LLM): deterministic, zero-latency, zero-cost. The patterns
cover the 80% of phrasings; if a user phrases it oddly the structured-param
path (API or sidebar) still works.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# State name → USPS code mapping (subset; cover the ones in our CSV).
_STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}

_DOLLAR_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)")


@dataclass
class ExtractedParams:
    acv: Optional[float] = None
    repair_cost: Optional[float] = None
    state_code: Optional[str] = None


def _find_dollar_near(text: str, keywords: list[str], window: int = 60) -> Optional[float]:
    """Find the first $amount that appears within `window` chars of any keyword.

    WHY proximity-based: "ACV is $8,000 and repair is $6,500" has two dollar
    amounts; we need the ACV one tied to "acv" and the repair one tied to
    "repair". Naive ordering would mis-assign on phrasings like
    "repair cost of $6,500 against an ACV of $8,000".
    """
    lower = text.lower()
    best: Optional[tuple[int, float]] = None  # (distance, value)
    for m in _DOLLAR_RE.finditer(text):
        try:
            value = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        pos = m.start()
        for kw in keywords:
            for km in re.finditer(rf"\b{re.escape(kw)}\b", lower):
                dist = min(abs(pos - km.start()), abs(pos - km.end()))
                if dist <= window and (best is None or dist < best[0]):
                    best = (dist, value)
    return best[1] if best else None


def _find_state_code(text: str) -> Optional[str]:
    # Two-letter USPS code surrounded by word boundaries (uppercase only to
    # avoid false positives like "IL" inside "I'll").
    m = re.search(r"\b([A-Z]{2})\b", text)
    if m and m.group(1) in set(_STATE_NAMES.values()):
        return m.group(1)
    lower = text.lower()
    # Match longer names first (e.g. "new york" before "new").
    for name in sorted(_STATE_NAMES.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", lower):
            return _STATE_NAMES[name]
    return None


def extract_params(message: str) -> ExtractedParams:
    """Best-effort extraction of acv / repair_cost / state_code from chat text."""
    if not message:
        return ExtractedParams()
    acv = _find_dollar_near(message, ["acv", "actual cash value", "value"], window=40)
    repair = _find_dollar_near(
        message,
        ["repair", "estimate", "fix", "fixing", "damage estimate", "repair cost"],
        window=40,
    )
    state = _find_state_code(message)
    return ExtractedParams(acv=acv, repair_cost=repair, state_code=state)
