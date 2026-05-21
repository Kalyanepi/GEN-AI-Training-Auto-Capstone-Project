"""Pre-agent input guardrails: PII, prompt injection, off-topic, jailbreak.

WHY pre-agent: blocking PII before the LLM ever sees it prevents the model
from echoing it back in logs, traces, or responses. Blocking jailbreaks before
the router prevents wasted tool invocations and protects the persona.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

from api.config import settings
from observability.logger import get_logger

logger = get_logger(__name__)


# WHY explicit block codes: API contract returns guardrail_reason; UI uses code
# to render appropriate message styling and adjuster contact details.
PII_DETECTED = "PII_DETECTED"
PROMPT_INJECTION = "PROMPT_INJECTION"
JAILBREAK_ATTEMPT = "JAILBREAK_ATTEMPT"
OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass
class GuardrailDecision:
    blocked: bool
    reason: Optional[str] = None
    message: Optional[str] = None


# Patterns ordered by specificity; first match wins.
_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "Credit card"),
    (re.compile(r"\b(0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])[/-](19|20)\d{2}\b"), "DOB"),
]

_JAILBREAK_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(?:different|general|unrestricted|dan)", re.IGNORECASE),
    re.compile(r"\bpretend\s+to\s+be\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(?:a\s+)?(?!an?\s+(?:adjuster|agent|policyholder))", re.IGNORECASE),
    re.compile(r"\bdo\s+anything\s+now\b|\bDAN\b", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
]

_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
    re.compile(r"</?\s*system\s*>", re.IGNORECASE),
    re.compile(r"reveal\s+(your|the)\s+(system\s+)?prompt", re.IGNORECASE),
]


_OUT_OF_SCOPE_CLASSIFIER_PROMPT = (
    "You are a binary classifier. Determine if the user message relates to "
    "auto insurance: coverage, claims, repairs, total loss, FNOL, rental, "
    "roadside, deductibles, policy questions. "
    "Reply with exactly one word: YES or NO.\n\n"
    "Examples:\n"
    "User: How much does front bumper replacement cost on a Mid-size Sedan? -> YES\n"
    "User: Repair estimate for airbag deployment in a Luxury SUV -> YES\n"
    "User: Will my car be totaled? ACV is $18,000 and repair cost is $14,000 -> YES\n"
    "User: How do I file a claim after hail damage? -> YES\n"
    "User: What is my rental daily limit? -> YES\n"
    "User: What is collision coverage? -> YES\n"
    "User: What is the weather today? -> NO\n"
    "User: Explain quantum mechanics -> NO"
)


def _safe_response(reason: str) -> str:
    """Standard guardrail response template per architecture plan §11.4."""
    return (
        "I can only help with auto insurance topics: policy coverage, repair "
        "estimates, total loss thresholds, FNOL guidance, rental limits, and "
        "roadside coverage.\n\n"
        f"For anything else, contact RoadGuard Claims: {settings.adjuster_phone} "
        f"or {settings.adjuster_url}."
    )


def _check_pii(message: str) -> Optional[GuardrailDecision]:
    for pattern, label in _PII_PATTERNS:
        if pattern.search(message):
            logger.warning("input_guardrail_pii", label=label)
            return GuardrailDecision(
                blocked=True,
                reason=PII_DETECTED,
                message=(
                    "For your security, please don't share personal identifiers "
                    "like SSN, credit card, or date of birth in chat. "
                    f"Contact your adjuster directly at {settings.adjuster_phone}."
                ),
            )
    return None


def _check_jailbreak(message: str) -> Optional[GuardrailDecision]:
    for pattern in _JAILBREAK_PATTERNS:
        if pattern.search(message):
            logger.warning("input_guardrail_jailbreak", pattern=pattern.pattern)
            return GuardrailDecision(
                blocked=True,
                reason=JAILBREAK_ATTEMPT,
                message=(
                    "I'm RoadGuard AI Copilot — I can only help with auto "
                    "insurance questions. " + _safe_response(JAILBREAK_ATTEMPT)
                ),
            )
    return None


def _check_prompt_injection(message: str) -> Optional[GuardrailDecision]:
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(message):
            logger.warning("input_guardrail_injection", pattern=pattern.pattern)
            return GuardrailDecision(
                blocked=True,
                reason=PROMPT_INJECTION,
                message=_safe_response(PROMPT_INJECTION),
            )
    return None


async def _check_out_of_scope(message: str, client: AsyncOpenAI) -> Optional[GuardrailDecision]:
    """LLM-based topical filter — the soft layer after fast regex checks.

    WHY LLM here: regex can't reliably catch "what's the weather?" or
    "explain quantum mechanics" — semantic relevance needs a model.
    """
    try:
        resp = await client.chat.completions.create(
            model=settings.openai_guardrail_model,
            messages=[
                {"role": "system", "content": _OUT_OF_SCOPE_CLASSIFIER_PROMPT},
                {"role": "user", "content": message},
            ],
            temperature=0.0,
            max_tokens=4,
        )
        verdict = (resp.choices[0].message.content or "").strip().upper()
        if verdict.startswith("NO"):
            logger.info("input_guardrail_off_topic")
            return GuardrailDecision(
                blocked=True,
                reason=OUT_OF_SCOPE,
                message=_safe_response(OUT_OF_SCOPE),
            )
    except Exception as e:
        # WHY fail-open on classifier errors: blocking on transient OpenAI
        # outages would produce a worse user experience than a single
        # potentially off-topic answer slipping through (downstream synthesis
        # will still ground in retrieved chunks or refuse).
        logger.warning("input_guardrail_classifier_error", error=str(e))
    return None


# Fast insurance-keyword bypass: if the message contains obvious auto-insurance
# vocabulary, skip the expensive/slow LLM classifier entirely.
# WHY: GPT-4o-mini sometimes misclassifies short insurance questions as
# off-topic (e.g., "Are intentional acts covered?"). A regex check is
# deterministic, instant, and avoids wasting an LLM call.
_INSURANCE_KEYWORDS_RE = re.compile(
    r"\b(?:"
    r"coverage|covered|cover|policy|claim|deductible|collision|comprehensive|"
    r"liability|rental|roadside|tow|total loss|repair|estimate|cost|fnol|"
    r"accident|damage|vehicle|car|auto|insured|adjuster|premium|tier|"
    r"exclusion|intentional|gap|medpay|uninsured|underinsured|um|uim|"
    r"windshield|bumper|airbag|hail|flood|theft|vandalism|fire|flood|"
    r"breakdown|lockout|jump|flat tire|replacement|acv|actual cash value|"
    r"oems?|aftermarket|depreciation|salvage|totaled|declare|file|report|"
    r"incident|loss|benefit|limit|schedule|declaration|exclusion|waiver|"
    r"glass|towing|mile|mileage|reimbursement|daily limit|days|"
    r"new car|rideshare|delivery|driver|passenger|third party|bodily|property|"
    r"injury|medical|expense|hospital|ambulance|legal|fault|negligence|"
    r"roadguard|insurance|insurer|underwriter"
    r")\b",
    re.IGNORECASE,
)


def _check_insurance_keywords(message: str) -> Optional[GuardrailDecision]:
    if _INSURANCE_KEYWORDS_RE.search(message):
        logger.info("input_guardrail_keyword_bypass")
        return GuardrailDecision(blocked=False)
    return None


async def check_input(message: str, client: Optional[AsyncOpenAI] = None) -> GuardrailDecision:
    """Run all input guardrails in fast-to-slow order. First blocker wins."""
    client = client or AsyncOpenAI(api_key=settings.openai_api_key)

    for fast_check in (_check_pii, _check_jailbreak, _check_prompt_injection):
        decision = fast_check(message)
        if decision and decision.blocked:
            return decision

    # Fast keyword bypass: if obvious insurance terms present, skip LLM.
    decision = _check_insurance_keywords(message)
    if decision:
        return decision

    decision = await _check_out_of_scope(message, client)
    if decision and decision.blocked:
        return decision

    return GuardrailDecision(blocked=False)
