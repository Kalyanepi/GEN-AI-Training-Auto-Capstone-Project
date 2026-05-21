"""Latency badge, tools-used row, and LangSmith trace link.

WHY a dedicated panel: makes the demo self-explanatory — judges can see at a
glance which tools fired and how fast (the 4s SLA from §12.3).
"""
from __future__ import annotations

from typing import List, Optional

import streamlit as st


def _latency_color(latency_ms: int) -> str:
    if latency_ms < 2000:
        return "green"
    if latency_ms < 4000:
        return "orange"
    return "red"


def render_metrics(
    latency_ms: int,
    tools_used: List[str],
    intent: Optional[str],
    trace_url: Optional[str],
    guardrail_reason: Optional[str] = None,
) -> None:
    color = _latency_color(latency_ms)
    cols = st.columns([1, 2, 2, 1])
    with cols[0]:
        st.markdown(f":{color}[**{latency_ms} ms**]")
    with cols[1]:
        if intent:
            st.markdown(f"**Intent:** `{intent}`")
    with cols[2]:
        if tools_used:
            st.markdown("**Tools:** " + " ".join(f"`{t}`" for t in tools_used))
    with cols[3]:
        if trace_url:
            st.markdown(f"[Trace ↗]({trace_url})")
    if guardrail_reason:
        st.warning(f"Guardrail: {guardrail_reason}", icon="⚠️")
