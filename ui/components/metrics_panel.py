"""Metrics row — latency pill, intent badge, tool chips, trace link."""
from __future__ import annotations

from typing import List, Optional

import streamlit as st


# Human-readable tool names for the demo audience.
_TOOL_LABELS = {
    "policy_rag_tool":          "Policy RAG",
    "repair_cost_tool":         "Repair Cost CSV",
    "total_loss_tool":          "Total Loss Calc",
    "fnol_guide_tool":          "FNOL Guide",
    "rental_lookup_tool":       "Rental Lookup",
    "roadside_tool":            "Roadside",
    "um_uim_tool":              "UM/UIM",
    "coverage_identifier_tool": "Coverage ID",
}

# Intent → friendly label.
_INTENT_LABELS = {
    "COVERAGE_QA":      "Coverage Q&A",
    "REPAIR_ESTIMATE":  "Repair Estimate",
    "TOTAL_LOSS":       "Total Loss",
    "FNOL_GUIDANCE":    "FNOL Guidance",
    "RENTAL_LOOKUP":    "Rental Lookup",
    "ROADSIDE":         "Roadside",
    "UM_UIM":           "UM/UIM",
    "MULTI_INTENT":     "Multi-Intent",
    "OUT_OF_SCOPE":     "Out of Scope",
}

_GUARDRAIL_LABELS = {
    "PII_DETECTED":        "PII Detected",
    "PROMPT_INJECTION":    "Injection Blocked",
    "JAILBREAK_ATTEMPT":   "Jailbreak Blocked",
    "OUT_OF_SCOPE":        "Off-Topic",
    "LEGAL_ADVICE":        "Legal Advice Blocked",
    "FAULT_DETERMINATION": "Fault Determination Blocked",
    "MISSING_CITATION":    "Citation Required",
    "FABRICATED_DATA":     "Unverified Data Blocked",
}


def _latency_pill(ms: int) -> str:
    if ms < 2000:
        css, label = "rg-pill-green", f"✓ {ms} ms"
    elif ms < 4000:
        css, label = "rg-pill-orange", f"⚡ {ms} ms"
    else:
        css, label = "rg-pill-red", f"⏱ {ms} ms"
    return f"<span class='rg-pill {css}'>{label}</span>"


def _tool_pill(tool_name: str) -> str:
    label = _TOOL_LABELS.get(tool_name, tool_name)
    return f"<span class='rg-pill rg-pill-blue'>{label}</span>"


def _intent_pill(intent: str) -> str:
    label = _INTENT_LABELS.get(intent, intent)
    return f"<span class='rg-pill rg-pill-gray'>{label}</span>"


def render_metrics(
    latency_ms: int,
    tools_used: List[str],
    intent: Optional[str],
    trace_url: Optional[str],
    guardrail_reason: Optional[str] = None,
) -> None:
    parts: List[str] = []

    # Latency.
    if latency_ms:
        parts.append(_latency_pill(latency_ms))

    # Intent.
    if intent and intent != "OUT_OF_SCOPE":
        parts.append(_intent_pill(intent))

    # Tools fired.
    for t in tools_used:
        parts.append(_tool_pill(t))

    # Trace link — show even if URL may be stale; useful in development.
    if trace_url:
        parts.append(
            f"<a href='{trace_url}' target='_blank' "
            f"style='font-size:0.75rem; color:#2980b9; text-decoration:none;'>"
            f"↗ LangSmith Trace</a>"
        )

    if parts:
        st.markdown(
            "<div class='rg-metrics'>" + "&nbsp; ".join(parts) + "</div>",
            unsafe_allow_html=True,
        )

    # Guardrail badge — only show if blocked.
    if guardrail_reason:
        label = _GUARDRAIL_LABELS.get(guardrail_reason, guardrail_reason)
        st.markdown(
            f"<div style='font-size:0.75rem; color:#a04000; margin-top:4px;'>"
            f"⚠️ Guardrail triggered: <strong>{label}</strong></div>",
            unsafe_allow_html=True,
        )
