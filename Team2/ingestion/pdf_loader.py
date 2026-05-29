"""Section-aware PDF chunking — two-pass strategy.

Pass 1: Detect section/subsection/part boundaries via regex. Each detected
        section becomes a candidate chunk preserving full semantic context.
Pass 2: For sections exceeding chunk_size_tokens, recursively split on
        paragraph/sentence boundaries with overlap.

WHY two-pass: A single recursive split on tokens would slice mid-sentence and
break references like "as defined in subsection 3.1". Section-aware Pass 1
keeps subsections intact; Pass 2 only kicks in for oversized sections.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import pypdf
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from api.config import settings
from observability.logger import get_logger

logger = get_logger(__name__)


# WHY: tiktoken's cl100k_base matches OpenAI embedding tokenization, so token
# counts here reflect what the embedder will actually see.
_ENCODER = tiktoken.get_encoding("cl100k_base")

# Section header patterns ordered most-specific-first.
# WHY: Insurance docs use a mix of "SECTION N: Title", "PART N - Title",
# "N.N Subsection", and ALL-CAPS headings. Catch all common forms.
_SECTION_PATTERNS = [
    re.compile(r"^\s*(SECTION\s+\d+[A-Z]?)[\s:.\-]+(.+?)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*(PART\s+[IVX\d]+)[\s:.\-]+(.+?)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*(\d+\.\d+(?:\.\d+)?)\s+([A-Z][^\n]{3,80})$", re.MULTILINE),
    re.compile(r"^\s*(ARTICLE\s+[IVX\d]+)[\s:.\-]+(.+?)$", re.IGNORECASE | re.MULTILINE),
]

@dataclass
class RawPage:
    """One page extracted from a PDF before chunking."""
    page_number: int
    text: str

@dataclass
class PdfChunk:
    """A semantically meaningful chunk ready for metadata tagging + embedding."""
    text: str
    source_file: str
    page_number: int
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    chunk_index: int = 0
    token_count: int = 0
    extra: dict = field(default_factory=dict)


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def extract_pages(pdf_path: Path) -> List[RawPage]:
    """Extract text page-by-page.

    WHY page granularity: we need accurate page_number metadata for citations.
    Concatenating then guessing pages breaks the "Page 4" citation guarantee.
    """
    reader = pypdf.PdfReader(str(pdf_path))
    pages: List[RawPage] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Normalize whitespace but preserve newlines (used by section regex).
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append(RawPage(page_number=i + 1, text=text.strip()))
    return pages


def _find_section_boundaries(text: str) -> List[tuple[int, str, str]]:
    """Return list of (start_offset, section_label, section_title) in order.

    WHY: We try multiple patterns and merge — declaration pages use different
    heading conventions than the full policy.
    """
    boundaries: List[tuple[int, str, str]] = []
    for pattern in _SECTION_PATTERNS:
        for m in pattern.finditer(text):
            label = m.group(1).strip()
            title = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""
            boundaries.append((m.start(), label, title))
    # Sort by offset and de-duplicate on offset (multiple regex hits at same spot).
    boundaries.sort(key=lambda b: b[0])
    deduped: List[tuple[int, str, str]] = []
    last_offset = -1
    for b in boundaries:
        if b[0] != last_offset:
            deduped.append(b)
            last_offset = b[0]
    return deduped


def _split_oversized_section(
    text: str,
    section_label: Optional[str],
    section_title: Optional[str],
    page_number: int,
    source_file: str,
    chunk_index_start: int,
) -> List[PdfChunk]:
    """Pass 2: recursive character splitter for sections too large for one chunk.

    WHY RecursiveCharacterTextSplitter over naive token slicing: it respects
    paragraph and sentence boundaries, so chunks remain readable and don't
    lose context across slice points.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        length_function=_count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    return [
        PdfChunk(
            text=p,
            source_file=source_file,
            page_number=page_number,
            section_title=section_title,
            subsection_title=section_label if section_label != section_title else None,
            chunk_index=chunk_index_start + i,
            token_count=_count_tokens(p),
        )
        for i, p in enumerate(pieces)
        if p.strip()
    ]


def chunk_pdf(pdf_path: Path) -> List[PdfChunk]:
    """Two-pass section-aware chunking of a single PDF."""
    source_file = pdf_path.name
    pages = extract_pages(pdf_path)
    chunks: List[PdfChunk] = []
    chunk_idx = 0

    for page in pages:
        if not page.text.strip():
            continue

        boundaries = _find_section_boundaries(page.text)

        if not boundaries:
            # No detected sections: treat the whole page as one section, let
            # Pass 2 split if oversized.
            section_chunks = _split_oversized_section(
                text=page.text,
                section_label=None,
                section_title=None,
                page_number=page.page_number,
                source_file=source_file,
                chunk_index_start=chunk_idx,
            )
            chunks.extend(section_chunks)
            chunk_idx += len(section_chunks)
            continue

        # Slice text by boundary offsets.
        boundaries.append((len(page.text), "", ""))  # sentinel end
        for i in range(len(boundaries) - 1):
            start, label, title = boundaries[i]
            end, _, _ = boundaries[i + 1]
            section_text = page.text[start:end].strip()
            if not section_text:
                continue
            tokens = _count_tokens(section_text)
            if tokens <= settings.chunk_size_tokens:
                chunks.append(
                    PdfChunk(
                        text=section_text,
                        source_file=source_file,
                        page_number=page.page_number,
                        section_title=title or label,
                        subsection_title=label if title and label != title else None,
                        chunk_index=chunk_idx,
                        token_count=tokens,
                    )
                )
                chunk_idx += 1
            else:
                # Pass 2 for oversized sections.
                sub_chunks = _split_oversized_section(
                    text=section_text,
                    section_label=label,
                    section_title=title or label,
                    page_number=page.page_number,
                    source_file=source_file,
                    chunk_index_start=chunk_idx,
                )
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)

    logger.info(
        "pdf_chunked",
        source_file=source_file,
        pages=len(pages),
        chunks=len(chunks),
    )
    return chunks


def chunk_all_pdfs(pdf_dir: Path) -> Iterator[PdfChunk]:
    """Yield chunks from every PDF in the directory.

    WHY generator: memory-light when corpus grows.
    """
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("no_pdfs_found", pdf_dir=str(pdf_dir))
        return
    for pdf in pdfs:
        try:
            yield from chunk_pdf(pdf)
        except Exception as e:
            # WHY: one bad PDF must not abort the whole ingestion run.
            logger.error("pdf_chunk_failed", source_file=pdf.name, error=str(e), exc_info=True)
