"""Sidebar — brand, tier, real recents, API health, and session controls.

All session_state keys read/written here are owned by ui/utils/session_state.py.
"""
from __future__ import annotations

import streamlit as st

from ui.utils.api_client import clear_session_sync, health_sync
from ui.utils.session_state import reset_session

# (key, display name, deductible blurb, dot color emoji)
TIERS = [
    ("standard", "Standard Shield",    "Collision $1,000 · Comp $500", "⚪"),
    ("premium",  "Premium Guard",      "Collision $500 · Comp $250",   "🔵"),
    ("elite",    "Comprehensive Elite","Collision $250 · Comp $0",     "🟣"),
]

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="rg-sidebar-brand">
              <div class="rg-sidebar-mark"><i class="ti ti-shield-check"></i></div>
              <div class="rg-sidebar-brand-text">RoadGuard AI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Policy Tier ───────────────────────────────────────────────
        st.markdown('<span class="rg-s-label">Policy Tier</span>', unsafe_allow_html=True)
        tier_labels = {k: label for k, label, *_ in TIERS}
        tier_options = ["standard", "premium", "elite"]

        try:
            selected = st.pills(
                "tier",
                options=tier_options,
                format_func=lambda t: tier_labels[t],
                default=st.session_state.policy_tier,
                label_visibility="collapsed",
            )
            if selected:
                st.session_state.policy_tier = selected
        except AttributeError:
            # Fallback for older Streamlit versions
            cols = st.columns(3)
            for col, (key, label, *_rest) in zip(cols, TIERS):
                with col:
                    active = st.session_state.policy_tier == key
                    if st.button(label[:9], key=f"tier_{key}",
                                 type="primary" if active else "secondary",
                                 use_container_width=True):
                        st.session_state.policy_tier = key
                        st.rerun()

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

        # ── Recent Conversations ──────────────────────────────────────
        user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
        recent_items = [m["content"] for m in reversed(user_msgs)]
        st.markdown('<span class="rg-s-label">Recent Conversations</span>', unsafe_allow_html=True)
        st.markdown("<div class='rg-recent-list'>", unsafe_allow_html=True)
        if recent_items:
            for i, text in enumerate(recent_items):
                short = (text[:48] + "…") if len(text) > 48 else text
                active_class = " active" if i == 0 else ""
                st.markdown(
                    f"<div class='rg-recent-item{active_class}'>"
                    f"  <i class='ti ti-message rg-recent-icon'></i>"
                    f"  <span class='rg-recent-text'>{short}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div class='rg-recent-empty'>No conversations yet</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='rg-sidebar-bottom'>", unsafe_allow_html=True)
        health = health_sync()
        ok = health.get("status") in {"ok", "degraded"}
        status_label = "API Healthy" if ok else "API Offline"
        status_class = "ok" if ok else "bad"
        st.markdown(
            f"<div class='rg-health {status_class}'>"
            f"  <span class='rg-health-dot'></span>"
            f"  <span>{status_label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── New Session button ────────────────────────────────────────
        st.markdown('<div class="rg-clear-wrap">', unsafe_allow_html=True)
        if st.button("↺  New Session", use_container_width=True, key="new_session_btn"):
            try:
                clear_session_sync(st.session_state.session_id)
            except Exception:
                pass
            reset_session()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
