"""Chat panel — message list + input + per-turn rendering."""
from __future__ import annotations

import streamlit as st

from ui.components.citation_card import render_citations
from ui.components.metrics_panel import render_metrics
from ui.utils.api_client import chat_sync


def _build_payload(message: str) -> dict:
    return {
        "session_id": st.session_state.session_id,
        "message": message,
        "policy_tier": st.session_state.policy_tier,
        "coverage_type": st.session_state.coverage_type,
        "vehicle_category": st.session_state.vehicle_category,
        "state_code": st.session_state.state_code,
    }


def _escape_dollars(text: str) -> str:
    """Escape literal $ so Streamlit's markdown doesn't render them as LaTeX.

    WHY: Streamlit treats $...$ as math mode; an answer containing
    "$6,500 is 81.25% of $8,000" gets rendered as a corrupted equation.
    """
    return (text or "").replace("$", "\\$")


def _render_assistant_message(msg: dict) -> None:
    with st.chat_message("assistant"):
        st.markdown(_escape_dollars(msg["content"]))
        if msg.get("calculation_breakdown"):
            with st.expander("Calculation breakdown", expanded=False):
                st.code(msg["calculation_breakdown"], language="text")
        render_citations(msg.get("citations") or [])
        if msg.get("disclaimer"):
            st.caption(msg["disclaimer"])
        render_metrics(
            latency_ms=msg.get("latency_ms", 0),
            tools_used=msg.get("tools_used") or [],
            intent=msg.get("intent_detected"),
            trace_url=msg.get("trace_url"),
            guardrail_reason=msg.get("guardrail_reason"),
        )


def render_chat_panel() -> None:
    st.markdown("## Chat")

    # Render history.
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(_escape_dollars(msg["content"]))
        else:
            _render_assistant_message(msg)

    # Input.
    user_input = st.chat_input("Ask about coverage, repair costs, total loss, FNOL, rental, or roadside...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(_escape_dollars(user_input))

    # WHY no outer st.chat_message here: _render_assistant_message already
    # wraps its content in st.chat_message("assistant"). Nesting raises a
    # StreamlitAPIException ("Chat messages cannot nested inside other chat messages").
    with st.spinner("Thinking..."):
        try:
            resp = chat_sync(_build_payload(user_input))
        except Exception as e:
            err_msg = f"Error contacting API: {e}"
            with st.chat_message("assistant"):
                st.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
            return

    assistant_msg = {
        "role": "assistant",
        "content": resp.get("answer", "(no answer)"),
        "citations": resp.get("citations") or [],
        "latency_ms": resp.get("latency_ms", 0),
        "tools_used": resp.get("tools_used") or [],
        "intent_detected": resp.get("intent_detected"),
        "trace_url": resp.get("trace_url"),
        "guardrail_reason": resp.get("guardrail_reason"),
        "disclaimer": resp.get("disclaimer"),
        "calculation_breakdown": resp.get("calculation_breakdown"),
    }
    st.session_state.messages.append(assistant_msg)
    _render_assistant_message(assistant_msg)
