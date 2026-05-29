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


def extract_all_state_codes(text: str) -> list[str]:
    """Return ALL state codes found in the message, in order of appearance.

    WHY: multi-state comparison queries like "Florida vs Texas" contain two
    states. Returning only the first one causes the second to be silently
    dropped. This lets the orchestrator invoke total_loss_tool once per state.
    """
    if not text:
        return []
    valid_codes = set(_STATE_NAMES.values())
    found: list[str] = []
    seen: set[str] = set()

    # Build a list of (position, code) for all matches, then sort by position.
    hits: list[tuple[int, str]] = []

    # Two-letter uppercase codes.
    for m in re.finditer(r"\b([A-Z]{2})\b", text):
        code = m.group(1)
        if code in valid_codes and code not in seen:
            hits.append((m.start(), code))
            seen.add(code)

    # Full state names (longer names checked first to avoid partial matches).
    lower = text.lower()
    for name in sorted(_STATE_NAMES.keys(), key=len, reverse=True):
        m = re.search(rf"\b{re.escape(name)}\b", lower)
        if m:
            code = _STATE_NAMES[name]
            if code not in seen:
                hits.append((m.start(), code))
                seen.add(code)

    hits.sort(key=lambda x: x[0])
    found = [code for _, code in hits]
    return found


_BARE_NUMBER_RE = re.compile(
    r"^\s*\$?\s*(\d[\d,]*(?:\.\d+)?)\s*\$?\s*(?:dollars?|usd|k|thousand)?\s*[.,!?]?\s*$",
    re.IGNORECASE,
)


def extract_bare_number(message: str) -> Optional[float]:
    """Extract a bare dollar amount from a short clarification answer.

    WHY: when the agent asks "What is the ACV?" and the user replies "4000"
    or "4000$" or "$4,000", _find_dollar_near won't match because there's no
    keyword nearby. This handles the bare-number case.
    Returns None if the message has more content than just a number.
    """
    m = _BARE_NUMBER_RE.match(message.strip())
    if not m:
        return None
    try:
        raw = m.group(1).replace(",", "")
        val = float(raw)
        # Handle "k" suffix: "4k" → 4000
        if re.search(r"k\b", message, re.IGNORECASE) and val < 1000:
            val *= 1000
        return val if val > 0 else None
    except ValueError:
        return None


def extract_params(message: str, missing_acv: bool = False, missing_repair: bool = False) -> ExtractedParams:
    """Best-effort extraction of acv / repair_cost / state_code from chat text.

    WHY missing_acv / missing_repair: when the agent just asked for a specific
    value, a bare-number reply ("4000", "4000$") should be assigned to that
    field even without a keyword nearby.
    """
    if not message:
        return ExtractedParams()
    acv = _find_dollar_near(message, ["acv", "actual cash value", "value"], window=60)
    repair = _find_dollar_near(
        message,
        ["repair", "repairs", "estimate", "fix", "fixing", "damage", "damage estimate", "repair cost", "costs", "needs"],
        window=60,
    )
    state = _find_state_code(message)

    # If neither keyword-based extraction found anything and there's a bare number,
    # assign it to whichever field is missing.
    if acv is None and repair is None:
        bare = extract_bare_number(message)
        if bare is not None:
            if missing_acv and not missing_repair:
                acv = bare
            elif missing_repair and not missing_acv:
                repair = bare
            elif missing_acv:
                # Both missing — can't determine which; leave for router to ask
                acv = bare

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


def extract_damage_types(message: str) -> list[str]:
    """Extract ALL distinct damage phrases from the message.

    WHY a list: real users describe multiple damages in one turn
    ("cracked headlight and hood dent"). Returning only one would skip
    the others. The tool layer fans out CSV lookups for each and the
    synthesis LLM aggregates the results.
    """
    if not message:
        return []
    matches = _DAMAGE_RE.findall(message)
    if not matches:
        return []
    seen = set()
    out: list[str] = []
    for m in matches:
        s = m.strip().lower()
        if s and s not in seen:
            seen.add(s)
            out.append(m.strip())
    return out


# ---------------------------------------------------------------------------
# Vehicle make/model → CSV vehicle_category mapper
# ---------------------------------------------------------------------------
#
# WHY a static dictionary (not an LLM): users say "Honda Civic" but the
# RepairCost CSV uses categories like "Economy/Compact". A small lookup
# table covers ~90% of common US vehicles deterministically with zero
# latency / cost. Unknown makes fall through and the tool aggregates
# across all categories — still better than refusing.
#
# Categories MUST match exact strings in RepairCost_ReferenceTable.csv:
#   Economy/Compact, Mid-size Sedan, Full-size Sedan, Luxury Sedan,
#   Compact SUV/Crossover, Mid-size SUV, Full-size SUV/Truck, Luxury SUV
#
# Ordered: longer make+model phrases checked before single-word makes.

_VEHICLE_MODEL_MAP: dict[str, str] = {
    # ── Economy / Compact ──
    "honda civic": "Economy/Compact",
    "honda fit": "Economy/Compact",
    "toyota corolla": "Economy/Compact",
    "toyota yaris": "Economy/Compact",
    "nissan sentra": "Economy/Compact",
    "nissan versa": "Economy/Compact",
    "hyundai elantra": "Economy/Compact",
    "hyundai accent": "Economy/Compact",
    "kia forte": "Economy/Compact",
    "kia rio": "Economy/Compact",
    "mazda 3": "Economy/Compact",
    "mazda3": "Economy/Compact",
    "ford fiesta": "Economy/Compact",
    "ford focus": "Economy/Compact",
    "chevrolet sonic": "Economy/Compact",
    "chevrolet spark": "Economy/Compact",
    "chevrolet cruze": "Economy/Compact",
    "volkswagen jetta": "Economy/Compact",
    "vw jetta": "Economy/Compact",
    "subaru impreza": "Economy/Compact",
    "mitsubishi mirage": "Economy/Compact",
    # ── Mid-size Sedan ──
    "honda accord": "Mid-size Sedan",
    "toyota camry": "Mid-size Sedan",
    "nissan altima": "Mid-size Sedan",
    "hyundai sonata": "Mid-size Sedan",
    "kia optima": "Mid-size Sedan",
    "kia k5": "Mid-size Sedan",
    "mazda 6": "Mid-size Sedan",
    "mazda6": "Mid-size Sedan",
    "ford fusion": "Mid-size Sedan",
    "chevrolet malibu": "Mid-size Sedan",
    "subaru legacy": "Mid-size Sedan",
    "volkswagen passat": "Mid-size Sedan",
    "vw passat": "Mid-size Sedan",
    # ── Full-size Sedan ──
    "toyota avalon": "Full-size Sedan",
    "nissan maxima": "Full-size Sedan",
    "chrysler 300": "Full-size Sedan",
    "dodge charger": "Full-size Sedan",
    "chevrolet impala": "Full-size Sedan",
    "ford taurus": "Full-size Sedan",
    "kia cadenza": "Full-size Sedan",
    "hyundai azera": "Full-size Sedan",
    # ── Luxury Sedan ──
    "bmw 3 series": "Luxury Sedan",
    "bmw 5 series": "Luxury Sedan",
    "bmw 7 series": "Luxury Sedan",
    "mercedes c-class": "Luxury Sedan",
    "mercedes e-class": "Luxury Sedan",
    "mercedes s-class": "Luxury Sedan",
    "mercedes-benz c-class": "Luxury Sedan",
    "mercedes-benz e-class": "Luxury Sedan",
    "audi a3": "Luxury Sedan",
    "audi a4": "Luxury Sedan",
    "audi a6": "Luxury Sedan",
    "audi a8": "Luxury Sedan",
    "lexus is": "Luxury Sedan",
    "lexus es": "Luxury Sedan",
    "lexus gs": "Luxury Sedan",
    "lexus ls": "Luxury Sedan",
    "infiniti q50": "Luxury Sedan",
    "infiniti q70": "Luxury Sedan",
    "acura tlx": "Luxury Sedan",
    "acura rlx": "Luxury Sedan",
    "cadillac cts": "Luxury Sedan",
    "cadillac ats": "Luxury Sedan",
    "tesla model 3": "Luxury Sedan",
    "tesla model s": "Luxury Sedan",
    # ── Compact SUV/Crossover ──
    "honda cr-v": "Compact SUV/Crossover",
    "honda crv": "Compact SUV/Crossover",
    "honda hr-v": "Compact SUV/Crossover",
    "toyota rav4": "Compact SUV/Crossover",
    "toyota rav-4": "Compact SUV/Crossover",
    "nissan rogue": "Compact SUV/Crossover",
    "mazda cx-5": "Compact SUV/Crossover",
    "mazda cx5": "Compact SUV/Crossover",
    "ford escape": "Compact SUV/Crossover",
    "chevrolet equinox": "Compact SUV/Crossover",
    "subaru forester": "Compact SUV/Crossover",
    "subaru crosstrek": "Compact SUV/Crossover",
    "hyundai tucson": "Compact SUV/Crossover",
    "kia sportage": "Compact SUV/Crossover",
    "jeep compass": "Compact SUV/Crossover",
    "jeep renegade": "Compact SUV/Crossover",
    # ── Mid-size SUV ──
    "honda pilot": "Mid-size SUV",
    "honda passport": "Mid-size SUV",
    "toyota highlander": "Mid-size SUV",
    "toyota 4runner": "Mid-size SUV",
    "nissan murano": "Mid-size SUV",
    "nissan pathfinder": "Mid-size SUV",
    "ford explorer": "Mid-size SUV",
    "ford edge": "Mid-size SUV",
    "chevrolet traverse": "Mid-size SUV",
    "chevrolet blazer": "Mid-size SUV",
    "hyundai santa fe": "Mid-size SUV",
    "kia sorento": "Mid-size SUV",
    "kia telluride": "Mid-size SUV",
    "mazda cx-9": "Mid-size SUV",
    "subaru ascent": "Mid-size SUV",
    "jeep grand cherokee": "Mid-size SUV",
    "jeep wrangler": "Mid-size SUV",
    # ── Full-size SUV/Truck ──
    "ford f-150": "Full-size SUV/Truck",
    "ford f150": "Full-size SUV/Truck",
    "ford f-250": "Full-size SUV/Truck",
    "ford f-350": "Full-size SUV/Truck",
    "ford expedition": "Full-size SUV/Truck",
    "chevrolet silverado": "Full-size SUV/Truck",
    "chevrolet tahoe": "Full-size SUV/Truck",
    "chevrolet suburban": "Full-size SUV/Truck",
    "gmc sierra": "Full-size SUV/Truck",
    "gmc yukon": "Full-size SUV/Truck",
    "ram 1500": "Full-size SUV/Truck",
    "ram 2500": "Full-size SUV/Truck",
    "dodge ram": "Full-size SUV/Truck",
    "toyota tundra": "Full-size SUV/Truck",
    "toyota sequoia": "Full-size SUV/Truck",
    "nissan titan": "Full-size SUV/Truck",
    "nissan armada": "Full-size SUV/Truck",
    # ── Luxury SUV ──
    "bmw x3": "Luxury SUV",
    "bmw x5": "Luxury SUV",
    "bmw x7": "Luxury SUV",
    "mercedes glc": "Luxury SUV",
    "mercedes gle": "Luxury SUV",
    "mercedes gls": "Luxury SUV",
    "mercedes g-class": "Luxury SUV",
    "audi q3": "Luxury SUV",
    "audi q5": "Luxury SUV",
    "audi q7": "Luxury SUV",
    "audi q8": "Luxury SUV",
    "lexus rx": "Luxury SUV",
    "lexus gx": "Luxury SUV",
    "lexus lx": "Luxury SUV",
    "lexus nx": "Luxury SUV",
    "infiniti qx50": "Luxury SUV",
    "infiniti qx60": "Luxury SUV",
    "infiniti qx80": "Luxury SUV",
    "acura rdx": "Luxury SUV",
    "acura mdx": "Luxury SUV",
    "cadillac escalade": "Luxury SUV",
    "cadillac xt5": "Luxury SUV",
    "porsche cayenne": "Luxury SUV",
    "porsche macan": "Luxury SUV",
    "tesla model x": "Luxury SUV",
    "tesla model y": "Luxury SUV",
    "land rover": "Luxury SUV",
    "range rover": "Luxury SUV",
}

# Brand-only fallback for unknown models. Conservative defaults.
_VEHICLE_BRAND_FALLBACK: dict[str, str] = {
    "bmw": "Luxury Sedan",
    "mercedes": "Luxury Sedan",
    "mercedes-benz": "Luxury Sedan",
    "audi": "Luxury Sedan",
    "lexus": "Luxury Sedan",
    "infiniti": "Luxury Sedan",
    "acura": "Luxury Sedan",
    "cadillac": "Luxury Sedan",
    "porsche": "Luxury Sedan",
    "tesla": "Luxury Sedan",
    "honda": "Mid-size Sedan",
    "toyota": "Mid-size Sedan",
    "nissan": "Mid-size Sedan",
    "hyundai": "Mid-size Sedan",
    "kia": "Mid-size Sedan",
    "mazda": "Mid-size Sedan",
    "ford": "Mid-size Sedan",
    "chevrolet": "Mid-size Sedan",
    "chevy": "Mid-size Sedan",
    "subaru": "Mid-size Sedan",
    "volkswagen": "Economy/Compact",
    "vw": "Economy/Compact",
    "jeep": "Compact SUV/Crossover",
    "ram": "Full-size SUV/Truck",
    "gmc": "Full-size SUV/Truck",
    "dodge": "Mid-size Sedan",
    "chrysler": "Mid-size Sedan",
    "mitsubishi": "Economy/Compact",
}


def extract_vehicle_category(message: str) -> Optional[str]:
    """Map a free-text vehicle reference to a CSV vehicle_category.

    Examples:
      "I drive a Honda Civic"   -> "Economy/Compact"
      "my BMW X5 needs repair"  -> "Luxury SUV"
      "2018 Toyota"             -> "Mid-size Sedan"  (brand fallback)
      "my car"                  -> None
    """
    if not message:
        return None
    lower = message.lower()
    # 1. Try most-specific make+model phrases first (longest first).
    for phrase in sorted(_VEHICLE_MODEL_MAP.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", lower):
            return _VEHICLE_MODEL_MAP[phrase]
    # 2. Brand-only fallback.
    for brand in sorted(_VEHICLE_BRAND_FALLBACK.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(brand)}\b", lower):
            return _VEHICLE_BRAND_FALLBACK[brand]
    return None
