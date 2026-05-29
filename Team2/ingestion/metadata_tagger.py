"""Assigns coverage_type, policy_tier, doc_type, and content flags per chunk.

WHY metadata-driven RAG: FAISS doesn't natively filter. We tag rich metadata at
ingestion time so the retriever can post-filter candidates by tier/coverage —
preventing the system from ever serving Elite deductibles ($250) to a Standard
user ($1,000).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List

from ingestion.pdf_loader import PdfChunk
from observability.logger import get_logger

logger = get_logger(__name__)


# WHY: filename-based doc_type mapping. The 9 PDFs have predictable names per
# the architecture plan §3, so we can deterministically tag them.
_DOC_TYPE_BY_FILE: Dict[str, str] = {
    "FullAutoPolicy_RoadGuard.pdf": "policy",
    "DeclarationPage_STANDARD_RoadGuard.pdf": "declaration",
    "DeclarationPage_PREMIUM_RoadGuard.pdf": "declaration",
    "DeclarationPage_ELITE_RoadGuard.pdf": "declaration",
    "FNOL_Intake_Guidelines_RoadGuard.pdf": "fnol",
    "RentalReimbursement_Schedule_RoadGuard.pdf": "rental",
    "UM_UIM_CoverageGuide_RoadGuard.pdf": "um_uim",
    "RoadsideAssistance_Towing_RoadGuard.pdf": "roadside",
    "AutoInsurance_Glossary_RoadGuard.pdf": "glossary",
}

# Tier inferred from declaration filename. Other docs default to all tiers.
_TIER_BY_FILE: Dict[str, List[str]] = {
    "DeclarationPage_STANDARD_RoadGuard.pdf": ["standard"],
    "DeclarationPage_PREMIUM_RoadGuard.pdf": ["premium"],
    "DeclarationPage_ELITE_RoadGuard.pdf": ["elite"],
}

# Keyword -> coverage_type list. A chunk can carry multiple coverage tags.
# WHY: a single "Section 3 Collision" paragraph may also mention deductibles
# (collision-only) AND mention comprehensive by contrast — tag both.
_COVERAGE_KEYWORDS: Dict[str, List[str]] = {
    "collision": ["collision", "rollover", "impact", "crash", "at-fault", "pothole"],
    "comprehensive": ["comprehensive", "theft", "vandalism", "hail", "flood", "fire", "deer", "weather", "glass", "windshield"],
    "liability": ["liability", "bodily injury", "property damage", "third party"],
    "um_uim": ["uninsured", "underinsured", "um/uim", "hit-and-run", "hit and run"],
    "gap": ["gap", "loan balance", "lease balance", "owe more"],
    "medpay": ["medpay", "medical payments", "med pay"],
    "rental": ["rental reimbursement", "rental car", "substitute vehicle"],
    "roadside": ["roadside", "tow", "towing", "jump start", "lockout", "winching", "flat tire"],
    "fnol": ["first notice of loss", "fnol", "claim filing", "report claim"],
}

# Tier hints inside body text (used for full policy / FNOL chunks that mention
# tier-specific clauses like "GAP coverage is included under Premium Guard and
# Comprehensive Elite only").
_TIER_KEYWORDS: Dict[str, List[str]] = {
    "standard": ["standard shield", "standard tier"],
    "premium": ["premium guard", "premium tier"],
    "elite": ["comprehensive elite", "elite tier"],
}


def _detect_coverage_types(text: str) -> List[str]:
    lower = text.lower()
    detected = [cov for cov, kws in _COVERAGE_KEYWORDS.items() if any(k in lower for k in kws)]
    return detected or ["general"]


def _detect_policy_tiers(text: str, source_file: str) -> List[str]:
    # Declaration pages take precedence — they ARE the tier definition.
    if source_file in _TIER_BY_FILE:
        return _TIER_BY_FILE[source_file]
    lower = text.lower()
    detected = [tier for tier, kws in _TIER_KEYWORDS.items() if any(k in lower for k in kws)]
    # WHY: if no tier explicitly mentioned, the clause applies to all tiers.
    return detected if detected else ["standard", "premium", "elite"]


def _content_flags(text: str) -> Dict[str, bool]:
    lower = text.lower()
    return {
        "contains_deductible_info": "deductible" in lower,
        "contains_exclusions": any(w in lower for w in ["exclusion", "not covered", "excluded", "does not cover"]),
        "contains_limits": any(w in lower for w in ["limit", "maximum", "per occurrence", "/day", "per term"]),
        "is_definition": bool(re.match(r"^\s*[A-Z][A-Za-z\s/\-]+:\s", text[:120])) or "means" in lower[:200],
    }


def _stable_chunk_id(source_file: str, page: int, section: str | None, text: str) -> str:
    """Deterministic, debuggable chunk_id.

    WHY: re-running ingestion should produce the same IDs so we can diff index
    contents and trace citation references in logs.
    """
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    base = source_file.replace(".pdf", "").lower()
    sec = re.sub(r"[^a-z0-9]+", "_", (section or "p").lower())[:24]
    return f"{base}_{sec}_p{page}_{h}"


def tag_chunk(chunk: PdfChunk) -> Dict:
    """Convert a PdfChunk into a fully-tagged metadata dict ready for the index."""
    doc_type = _DOC_TYPE_BY_FILE.get(chunk.source_file, "other")
    coverage_types = _detect_coverage_types(chunk.text)
    policy_tiers = _detect_policy_tiers(chunk.text, chunk.source_file)
    flags = _content_flags(chunk.text)
    chunk_id = _stable_chunk_id(chunk.source_file, chunk.page_number, chunk.section_title, chunk.text)

    return {
        "chunk_id": chunk_id,
        "text": chunk.text,
        "source_file": chunk.source_file,
        "page_number": chunk.page_number,
        "section_title": chunk.section_title,
        "subsection_title": chunk.subsection_title,
        "coverage_type": coverage_types,
        "policy_tier": policy_tiers,
        "doc_type": doc_type,
        "token_count": chunk.token_count,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        **flags,
    }


def tag_chunks(chunks: List[PdfChunk]) -> List[Dict]:
    tagged = [tag_chunk(c) for c in chunks]
    logger.info("metadata_tagged", total=len(tagged))
    return tagged
