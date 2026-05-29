"""Pre-agent input guardrails: PII (Presidio), prompt injection, off-topic, jailbreak.

WHY pre-agent: blocking PII before the LLM ever sees it prevents the model
from echoing it back in logs, traces, or responses. Blocking jailbreaks before
the router prevents wasted tool invocations and protects the persona.

WHY Presidio for PII: regex patterns miss names, addresses, and uncommon
formats. Presidio's NLP-backed recognizers (spaCy NER + rule-based) provide
coverage across 50+ PII entity types without manual pattern maintenance.
Custom recognizers extend it for insurance-domain identifiers (policy numbers,
claim IDs) that aren't in the default set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from openai import AsyncOpenAI
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider

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


# ---------------------------------------------------------------------------
# Presidio PII engine — lazy singleton so it loads once on first request.
# WHY lru_cache(1): AnalyzerEngine + spaCy model take ~2s to initialise;
# caching avoids that cost on every request.
# ---------------------------------------------------------------------------

def _make_presidio_engine() -> AnalyzerEngine:
    """Build and return a Presidio AnalyzerEngine with insurance-domain extras."""
    # Use the large spaCy model for best NER accuracy.
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    })
    nlp_engine = provider.create_engine()
    engine = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

    # Custom recognizer: RoadGuard policy numbers (POL-XXXXX or POL123456).
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="POLICY_NUMBER",
        patterns=[Pattern(
            name="policy_number",
            regex=r"\bPOL[-]?\d{4,10}\b",
            score=0.9,
        )],
    ))

    # Custom recognizer: claim IDs (CL-XXXX or CLM-XXXXX).
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="CLAIM_ID",
        patterns=[Pattern(
            name="claim_id",
            regex=r"\bCL(?:M)?[-]?\d{3,10}\b",
            score=0.9,
        )],
    ))

    # Custom recognizer: driver's license common US format (1-2 letters + 6-8 digits).
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="DRIVER_LICENSE",
        patterns=[Pattern(
            name="driver_license",
            regex=r"\b[A-Z]{1,2}\d{6,8}\b",
            score=0.6,
        )],
    ))

    # Custom recognizer: US SSN — Presidio's built-in requires context words
    # ("SSN", "social security") to score above 0.5. This explicit pattern
    # guarantees detection regardless of surrounding text.
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="US_SSN",
        patterns=[Pattern(
            name="us_ssn_explicit",
            regex=r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
            score=0.85,
        )],
    ))

    # Custom recognizer: US phone numbers in common formats.
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=[
            Pattern(
                name="us_phone_dashes",
                regex=r"\b\d{3}-\d{3}-\d{4}\b",
                score=0.75,
            ),
            Pattern(
                name="us_phone_parens",
                regex=r"\(\d{3}\)\s*\d{3}[-.\s]\d{4}\b",
                score=0.75,
            ),
            Pattern(
                name="us_phone_dots",
                regex=r"\b\d{3}\.\d{3}\.\d{4}\b",
                score=0.75,
            ),
        ],
    ))
    return engine


@lru_cache(maxsize=1)
def _get_presidio_engine() -> AnalyzerEngine:
    return _make_presidio_engine()


# PII entity types to block. We exclude LOCATION intentionally — city/state
# names are normal in insurance queries ("I'm in Texas").
_BLOCKED_PII_ENTITIES: List[str] = [
    # PERSON excluded — too noisy for insurance domain (car makes, adjuster
    # names, "John Doe" examples all trigger false positives).
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "US_BANK_NUMBER",
    "US_PASSPORT",
    "MEDICAL_LICENSE",
    "IP_ADDRESS",
    "IBAN_CODE",
    "POLICY_NUMBER",
    "CLAIM_ID",
    "DRIVER_LICENSE",
]
# Minimum confidence score to treat a detection as a true positive.
# 0.5 catches SSN (xxx-xx-xxxx) and phone numbers which Presidio scores
# at ~0.5-0.6. Insurance queries never contain these patterns so the
# false-positive risk is negligible.
_PII_SCORE_THRESHOLD = 0.5

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
        "roadside coverage."
    )


def _check_pii(message: str) -> Optional[GuardrailDecision]:
    """Presidio-powered PII detection with insurance-domain custom recognizers.

    WHY Presidio over regex: NLP-backed entity recognition catches names,
    addresses, and context-dependent identifiers that pure regex misses, while
    the custom recognizers add POL/CL patterns not in the default set.
    """
    try:
        engine = _get_presidio_engine()
        results = engine.analyze(
            text=message,
            language="en",
            entities=_BLOCKED_PII_ENTITIES,
            score_threshold=_PII_SCORE_THRESHOLD,
        )
        if results:
            detected = ", ".join(sorted({r.entity_type for r in results}))
            logger.warning("input_guardrail_pii_presidio", entities=detected)
            return GuardrailDecision(
                blocked=True,
                reason=PII_DETECTED,
                message=(
                    "For your security, please don't share personal identifiers "
                    "like your name, SSN, credit card, phone number, or policy/claim ID in chat."
                ),
            )
    except Exception as exc:
        # WHY fail-open: a Presidio/spaCy error should not block legitimate
        # insurance queries. Log and continue — downstream LLM synthesis will
        # still not echo any PII it encounters.
        logger.warning("input_guardrail_pii_error", error=str(exc))
    return None


def _check_jailbreak(message: str) -> Optional[GuardrailDecision]:
    for pattern in _JAILBREAK_PATTERNS:
        if pattern.search(message):
            logger.warning("input_guardrail_jailbreak", pattern=pattern.pattern)
            return GuardrailDecision(
                blocked=True,
                reason=JAILBREAK_ATTEMPT,
                message=(
                    "I'm Auto Insurance AI Copilot — I can only help with auto "
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
_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|howdy|greetings|good\s+(?:morning|afternoon|evening|day)|"
    r"what'?s\s+up|sup|hiya|yo|namaste|salut|hola)\b[\s!?.]*$",
    re.IGNORECASE,
)


def _check_greeting(message: str) -> bool:
    """Return True if the message is a pure greeting — let it pass through."""
    return bool(_GREETING_RE.match(message.strip()))


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
    r"roadguard|insurance|insurer|underwriter|"
    r"next steps?|what next|what now|what should i|what do i|how do i|"
    r"what happens|what are my|tell me more|explain|clarify|"
    r"steps?|process|procedure|timeline|deadline|how long|"
    r"should i|can i|do i need|is it|am i|will i|will my"
    r")\b",
    re.IGNORECASE,
)


def _check_insurance_keywords(message: str) -> Optional[GuardrailDecision]:
    if _INSURANCE_KEYWORDS_RE.search(message):
        logger.info("input_guardrail_keyword_bypass")
        return GuardrailDecision(blocked=False)
    return None


# Matches short clarification answers: dollar amounts (either end), plain
# numbers, state names/codes, or short phrases (≤40 chars).
_CLARIFICATION_ANSWER_RE = re.compile(
    r"^\s*("
    r"\$?\s*\d[\d,]*(?:\.\d+)?\s*\$?"  # $4000, 4000$, $4,500, 14000
    r"|\d+(?:\.\d+)?\s*(?:dollars?|usd|k|thousand|million)?"  # 4000 dollars, 4k
    r"|[A-Za-z]{2}(?:\s+[A-Za-z]+)*"  # TX, Florida, New York, yes, no
    r")\s*[.,!?]?\s*$",
    re.IGNORECASE,
)


def _check_clarification_answer(
    message: str, last_intent: Optional[str]
) -> Optional[GuardrailDecision]:
    """Bypass guardrail for short answers to clarification questions.

    WHY: when the agent asks "What is the ACV of your vehicle?" the user
    may reply with just "$4,000" or "4000$" or "Texas". These bare answers
    look off-topic to the LLM classifier but are valid insurance responses
    in context. We bypass when last_intent was CLARIFICATION_NEEDED.
    """
    if last_intent != "CLARIFICATION_NEEDED":
        return None
    stripped = message.strip()
    # Short messages (≤50 chars) that are answering a clarification — bypass.
    if len(stripped) <= 50 and _CLARIFICATION_ANSWER_RE.match(stripped):
        logger.info("input_guardrail_clarification_answer_bypass")
        return GuardrailDecision(blocked=False)
    # Also bypass very short messages (≤15 chars) unconditionally when
    # last_intent was CLARIFICATION_NEEDED — catches edge cases like "4000$".
    if len(stripped) <= 15:
        logger.info("input_guardrail_clarification_short_bypass")
        return GuardrailDecision(blocked=False)
    return None


async def check_input(
    message: str,
    client: Optional[AsyncOpenAI] = None,
    last_intent: Optional[str] = None,
) -> GuardrailDecision:
    """Run all input guardrails in fast-to-slow order. First blocker wins."""
    client = client or AsyncOpenAI(api_key=settings.openai_api_key)

    # Greetings always pass — the router will handle them with a welcome reply.
    if _check_greeting(message):
        logger.info("input_guardrail_greeting_bypass")
        return GuardrailDecision(blocked=False)

    # Short answers to clarification questions bypass LLM classifier.
    decision = _check_clarification_answer(message, last_intent)
    if decision:
        return decision

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
