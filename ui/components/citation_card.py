"""Citation cards — expandable, build trust by exposing exact source text."""
from __future__ import annotations

from typing import List

import streamlit as st


def render_citations(citations: List[dict]) -> None:
    if not citations:
        return
    st.markdown(f"**Citations ({len(citations)}):**")
    for i, c in enumerate(citations, start=1):
        doc = c.get("document", "Unknown source")
        section = c.get("section") or ""
        page = c.get("page")
        excerpt = c.get("excerpt", "")
        score = c.get("relevance_score", 0.0)
        source_type = c.get("source_type", "pdf")

        page_str = f", Page {page}" if page else ""
        icon = "📄" if source_type == "pdf" else "📊"
        title = f"{icon} [{i}] {doc}"
        if section:
            title += f" — {section}"
        title += f"{page_str}  ·  score {score:.2f}"

        with st.expander(title, expanded=False):
            # WHY escape $: Streamlit markdown interprets $...$ as LaTeX.
            safe_excerpt = (excerpt or "").replace("$", "\\$")
            st.markdown(f"> {safe_excerpt}")
            st.caption(f"chunk_id: `{c.get('chunk_id', '')}`  ·  source: {source_type}")
