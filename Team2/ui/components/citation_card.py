"""Citation cards — color-coded by source type, expandable with excerpt."""
from __future__ import annotations

from typing import List

import streamlit as st


# Friendly labels and colors per source type.
_SOURCE_META = {
    "pdf": {
        "icon":    "📄",
        "label":   "Policy PDF",
        "css":     "rg-citation-pdf",
        "color":   "#2980b9",
    },
    "csv": {
        "icon":    "📊",
        "label":   "Reference Data",
        "css":     "rg-citation-csv",
        "color":   "#27ae60",
    },
}
_DEFAULT_META = {"icon": "📎", "label": "Source", "css": "rg-citation-pdf", "color": "#7f8c8d"}


def _relevance_bar(score: float) -> str:
    """Return a compact visual bar string representing relevance (0–1)."""
    filled = round(score * 10)
    bar = "█" * filled + "░" * (10 - filled)
    pct = int(score * 100)
    return f"`{bar}` {pct}%"


def render_citations(citations: List[dict]) -> None:
    if not citations:
        return

    st.markdown(f"**Sources ({len(citations)}):**")

    for i, c in enumerate(citations, start=1):
        source_type = c.get("source_type", "pdf")
        meta = _SOURCE_META.get(source_type, _DEFAULT_META)

        doc      = c.get("document", "Unknown source")
        section  = c.get("section") or ""
        page     = c.get("page")
        excerpt  = c.get("excerpt", "")
        score    = float(c.get("relevance_score") or 0.0)
        chunk_id = c.get("chunk_id", "")

        # Build the expander title.
        page_str = f", p.{page}" if page else ""
        title = f"{meta['icon']} [{i}] {doc}"
        if section:
            title += f" — {section}"
        title += page_str

        with st.expander(title, expanded=False):
            # Color-coded excerpt block.
            safe_excerpt = (excerpt or "No excerpt available.").replace("$", "\\$")
            st.markdown(
                f"<div class='{meta['css']}'>"
                f"<em>{safe_excerpt}</em>"
                f"</div>",
                unsafe_allow_html=True,
            )
            # Relevance + metadata row.
            col1, col2 = st.columns([3, 2])
            with col1:
                st.caption(f"Relevance: {_relevance_bar(score)}")
            with col2:
                st.caption(
                    f"<span style='color:{meta['color']};font-weight:600;'>"
                    f"{meta['label']}</span>",
                    unsafe_allow_html=True,
                )
            if chunk_id:
                st.caption(f"chunk: `{chunk_id}`")
