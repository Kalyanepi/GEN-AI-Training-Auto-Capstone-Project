"""Drops empty/junk chunks and verifies metadata completeness before embedding.

WHY pre-embedding validation: every embedding API call costs money and time.
Junk chunks (page footers, "Page 4 of 12", header stamps, OCR noise) waste
both AND pollute retrieval results.
"""
from __future__ import annotations

import re
from typing import Dict, List

from observability.logger import get_logger

logger = get_logger(__name__)


_MIN_TOKENS = 8
_MIN_CHARS = 30
_REQUIRED_METADATA_KEYS = {
    "chunk_id",
    "text",
    "source_file",
    "page_number",
    "coverage_type",
    "policy_tier",
    "doc_type",
}

# WHY: footer/header noise patterns we want to discard wholesale.
_NOISE_PATTERNS = [
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*©.*roadguard.*$", re.IGNORECASE),
    re.compile(r"^\s*confidential\s*$", re.IGNORECASE),
]


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    return any(p.match(stripped) for p in _NOISE_PATTERNS)


def validate_chunks(chunks: List[Dict]) -> List[Dict]:
    """Return only chunks that pass quality + completeness checks."""
    kept: List[Dict] = []
    dropped_empty = 0
    dropped_short = 0
    dropped_noise = 0
    dropped_missing_meta = 0

    for c in chunks:
        text = (c.get("text") or "").strip()
        if not text:
            dropped_empty += 1
            continue
        if len(text) < _MIN_CHARS or c.get("token_count", 0) < _MIN_TOKENS:
            dropped_short += 1
            continue
        if _is_noise(text):
            dropped_noise += 1
            continue
        missing = _REQUIRED_METADATA_KEYS - set(c.keys())
        if missing:
            logger.warning("chunk_missing_metadata", missing=list(missing), chunk_id=c.get("chunk_id"))
            dropped_missing_meta += 1
            continue
        kept.append(c)

    logger.info(
        "chunk_validation_complete",
        kept=len(kept),
        dropped_empty=dropped_empty,
        dropped_short=dropped_short,
        dropped_noise=dropped_noise,
        dropped_missing_meta=dropped_missing_meta,
    )
    return kept
