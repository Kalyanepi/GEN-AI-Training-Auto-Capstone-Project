"""Chat panel — message list + input + per-turn rendering."""
from __future__ import annotations

import streamlit as st

from ui.components.citation_card import render_citations
from ui.components.metrics_panel import render_metrics
from ui.utils.api_client import chat_sync

# Guardrail reasons that warrant a warning-style display.
_GUARDRAIL_ICONS = {
    "PII_DETECTED":       "🔒",
    "PROMPT_INJECTION":   "🚫",
    "JAILBREAK_ATTEMPT":  "🚫",
    "OUT_OF_SCOPE":       "🔔",
    "LEGAL_ADVICE":       "⚖️",
    "FAULT_DETERMINATION":"⚖️",
    "MISSING_CITATION":   "📎",
    "FABRICATED_DATA":    "⚠️",
}


def _build_payload(message: str) -> dict:
    """Build the ChatRequest payload from current session state.

    WHY include acv/repair_cost/vehicle_year here: these come from sidebar
    number inputs — structured and reliable. The backend param_extractor still
    runs as a fallback for users who type them inline, but explicit sidebar
    values take precedence (see chat_service.py hydration logic).
    """
    payload: dict = {
        "session_id":       st.session_state.session_id,
        "message":          message,
        "policy_tier":      st.session_state.policy_tier,
        "coverage_type":    st.session_state.coverage_type,
        "vehicle_category": st.session_state.vehicle_category,
        "state_code":       st.session_state.state_code,
        "vehicle_year":     st.session_state.vehicle_year,
    }
    if st.session_state.acv:
        payload["acv"] = st.session_state.acv
    if st.session_state.repair_cost:
        payload["repair_cost"] = st.session_state.repair_cost
    return payload


def _escape_dollars(text: str) -> str:
    """Escape literal $ so Streamlit's markdown doesn't render them as LaTeX."""
    return (text or "").replace("$", "\\$")


def _render_assistant_message(msg: dict) -> None:
    guardrail = msg.get("guardrail_reason")

    with st.chat_message("assistant"):
        if guardrail:
            # Guardrail-blocked responses get a distinct visual treatment.
            icon = _GUARDRAIL_ICONS.get(guardrail, "⚠️")
            st.markdown(
                f"<div class='rg-guardrail'>{icon} {_escape_dollars(msg['content'])}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(_escape_dollars(msg["content"]))

        # Calculation breakdown (total loss tool).
        if msg.get("calculation_breakdown"):
            with st.expander("📊 Calculation breakdown", expanded=True):
                st.markdown(
                    f"<div class='rg-breakdown'>{msg['calculation_breakdown']}</div>",
                    unsafe_allow_html=True,
                )

        render_citations(msg.get("citations") or [])

        if msg.get("disclaimer") and not guardrail:
            st.markdown(
                f"<div class='rg-disclaimer'>ℹ️ {msg['disclaimer']}</div>",
                unsafe_allow_html=True,
            )

        render_metrics(
            latency_ms=msg.get("latency_ms", 0),
            tools_used=msg.get("tools_used") or [],
            intent=msg.get("intent_detected"),
            trace_url=msg.get("trace_url"),
            guardrail_reason=guardrail,
        )


def render_chat_panel() -> None:
    # Render history.
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(_escape_dollars(msg["content"]))
        else:
            _render_assistant_message(msg)

    # Input box.
    user_input = st.chat_input(
        "Ask about coverage, repair costs, total loss, FNOL, rental, or roadside..."
    )
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(_escape_dollars(user_input))

    with st.spinner("Searching policy documents and calculating..."):
        try:
            resp = chat_sync(_build_payload(user_input))
        except Exception as e:
            err_msg = f"Connection error: {e}"
            with st.chat_message("assistant"):
                st.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
            return

    assistant_msg = {
        "role":                  "assistant",
        "content":               resp.get("answer", "(no answer)"),
        "citations":             resp.get("citations") or [],
        "latency_ms":            resp.get("latency_ms", 0),
        "tools_used":            resp.get("tools_used") or [],
        "intent_detected":       resp.get("intent_detected"),
        "trace_url":             resp.get("trace_url"),
        "guardrail_reason":      resp.get("guardrail_reason"),
        "disclaimer":            resp.get("disclaimer"),
        "calculation_breakdown": resp.get("calculation_breakdown"),
    }
    st.session_state.messages.append(assistant_msg)
    _render_assistant_message(assistant_msg)
