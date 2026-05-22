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


# State name -> USPS code mapping (subset; cover the ones in our CSV).
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
    """Find the first $amount that appears within `window` chars of any keyword."""
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
    # Two-letter USPS code surrounded by word boundaries (uppercase only).
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


# ---------------------------------------------------------------------------
# Damage type extraction — used by repair_cost_tool
# ---------------------------------------------------------------------------

# Ordered from most specific (multi-word) to least to avoid short matches
# winning over descriptive ones (e.g. "rear bumper" before "bumper").
_DAMAGE_PARTS = [
    # Structural / frame
    r"frame\s+(?:straightening|damage|repair)",
    r"structural\s+(?:damage|repair)",
    # Airbag
    r"airbag\s+(?:deployment|replacement|repair)",
    r"dual\s+airbag",
    # Glass
    r"windshield\s+(?:replacement|crack|chip|repair)",
    r"rear\s+window\s+(?:replacement|damage)",
    r"side\s+window\s+(?:replacement|damage)",
    r"glass\s+(?:replacement|damage)",
    # Bumper
    r"(?:front|rear)\s+bumper\s+(?:replacement|repair|damage)",
    r"bumper\s+(?:replacement|repair|damage)",
    # Hood / trunk
    r"hood\s+(?:replacement|dent|damage|repair)",
    r"trunk\s+(?:replacement|damage|repair)",
    # Door / fender / quarter
    r"(?:door|fender|quarter\s+panel)\s+(?:replacement|dent|damage|repair)",
    # Headlights / taillights
    r"(?:headlight|taillight|tail\s+light|head\s+light)\s+(?:replacement|damage)",
    # Roof
    r"roof\s+(?:damage|replacement|repair)",
    # Engine / transmission
    r"engine\s+(?:damage|replacement|repair)",
    r"transmission\s+(?:damage|replacement|repair)",
    # Flood / fire / hail / theft / vandalism
    r"(?:flood|fire|hail|theft|vandalism)\s+damage",
    r"hail\s+(?:damage|dent)",
    r"water\s+damage",
    # Generic with location prefix
    r"(?:front|rear|side|left|right)\s+(?:bumper|door|fender|panel|damage)",
    # Fallback single words
    r"\b(?:bumper|hood|windshield|airbag|frame|fender|door|roof|engine|"
    r"headlight|taillight|trunk|quarter\s+panel|structural|hail|flood|fire|"
    r"theft|vandalism|glass|mirror|tire|wheel)\b",
]

_DAMAGE_RE = re.compile(
    "|".join(f"(?:{p})" for p in _DAMAGE_PARTS),
    re.IGNORECASE,
)


def extract_damage_type(message: str) -> Optional[str]:
    """Extract a short damage description suitable for repair_cost_tool fuzzy matching.

    WHY: repair_cost_tool's fuzzy matcher expects a short phrase like
    "rear bumper replacement" or "windshield crack", not a full sentence.
    Passing the raw user message gives SequenceMatcher a ~5% ratio, well
    below the 0.60 threshold, causing every repair lookup to fail silently.

    Returns the best matching phrase found, or None if nothing is recognizable.
    """
    if not message:
        return None
    # Find all matches, prefer the most specific (longest) one.
    matches = _DAMAGE_RE.findall(message)
    if not matches:
        return None
    # Filter empty strings from alternation groups, return longest match.
    candidates = [m for m in matches if m.strip()]
    if not candidates:
        return None
    return max(candidates, key=len).strip()
