"""Build Citation objects from retrieved chunks (and CSV-sourced answers).

WHY a separate module: citation rules (max 4, ranked by score, ≤150-char
excerpt, no fabrication below threshold) appear in multiple tools — keeping
the logic in one place prevents drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from api.config import settings
from rag.retriever import ChunkResult


@dataclass
class Citation:
    """Public-facing citation surfaced in API + UI."""
    document: str
    section: Optional[str]
    page: Optional[int]
    excerpt: str
    relevance_score: float
    chunk_id: str
    source_type: str  # "pdf" | "csv"


def _truncate_excerpt(text: str, max_chars: Optional[int] = None) -> str:
    """Trim chunk text to excerpt length with ellipsis.

    WHY trim at word boundary: hard-cut at char N produces ugly UI like
    "your collision deduct..." — better to back off to last space.
    """
    limit = max_chars or settings.citation_excerpt_max_chars
    text = " ".join(text.split())  # collapse whitespace
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut}..."


def chunks_to_citations(chunks: List[ChunkResult], max_citations: Optional[int] = None) -> List[Citation]:
    """Convert retrieved chunks into ranked, capped citations."""
    cap = max_citations or settings.max_citations_per_response
    sorted_chunks = sorted(chunks, key=lambda c: c.similarity_score, reverse=True)[:cap]
    return [
        Citation(
            document=c.source_file,
            section=c.section_title or c.subsection_title,
            page=c.page_number,
            excerpt=_truncate_excerpt(c.text),
            relevance_score=round(c.similarity_score, 3),
            chunk_id=c.chunk_id,
            source_type="pdf",
        )
        for c in sorted_chunks
    ]


def csv_citation(
    csv_filename: str,
    row_descriptor: str,
    excerpt: str,
    relevance_score: float = 1.0,
) -> Citation:
    """Build a citation for a structured CSV lookup result.

    WHY: total_loss_tool / repair_cost_tool answers must cite the specific CSV
    row used (e.g., 'TotalLoss_Threshold_Table.csv — Illinois 75% row'),
    matching the architecture plan §18 contract.
    """
    return Citation(
        document=csv_filename,
        section=row_descriptor,
        page=None,
        excerpt=_truncate_excerpt(excerpt),
        relevance_score=round(relevance_score, 3),
        chunk_id=f"csv:{csv_filename}:{row_descriptor}",
        source_type="csv",
    )
