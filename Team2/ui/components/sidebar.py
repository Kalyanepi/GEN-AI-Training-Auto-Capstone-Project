"""Sidebar — brand, tier, real recents, API health, and session controls.

All session_state keys read/written here are owned by ui/utils/session_state.py.
"""
from __future__ import annotations

import threading
import time
import streamlit as st

from ui.utils.api_client import clear_session_sync, health_sync
from ui.utils.session_state import reset_session


# ── Health check cache (avoid blocking every rerun) ───────────────────
_health_cache: dict = {"status": "ok", "ts": 0.0}
_HEALTH_TTL = 30.0  # seconds


def _get_health_cached() -> dict:
    now = time.monotonic()
    if now - _health_cache["ts"] > _HEALTH_TTL:
        result = health_sync()
        _health_cache.update({"status": result.get("status", "ok"), "ts": now})
    return _health_cache

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
              <div class="rg-sidebar-brand-text">Auto Insurance AI</div>
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
        health = _get_health_cached()
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

        # ── Developer Console button ──────────────────────────────────
        st.markdown("""
        <style>
        .rg-dev-wrap [data-testid="stButton"] > button {
          width: 100% !important; background: rgba(109,93,252,.12) !important;
          border: 1px solid rgba(109,93,252,.3) !important;
          border-radius: 12px !important; color: #a99cff !important;
          font-size: 12.5px !important; font-weight: 600 !important;
          padding: 9px 12px !important; transition: all .18s !important;
          margin-bottom: 8px !important;
        }
        .rg-dev-wrap [data-testid="stButton"] > button:hover {
          background: rgba(109,93,252,.22) !important;
          border-color: rgba(109,93,252,.55) !important;
          box-shadow: 0 0 14px rgba(109,93,252,.25) !important;
          color: #c4b8ff !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="rg-dev-wrap">', unsafe_allow_html=True)
        st.markdown(
            '<a href="/dev_console" target="_self" style="display:block;text-align:center;'
            'background:rgba(109,93,252,.12);border:1px solid rgba(109,93,252,.3);'
            'border-radius:12px;color:#a99cff;font-size:12.5px;font-weight:600;'
            'padding:9px 12px;text-decoration:none;margin-bottom:8px;'
            'transition:all .18s;">🔧  Developer Console</a>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── New Session button ────────────────────────────────────────
        st.markdown('<div class="rg-clear-wrap">', unsafe_allow_html=True)
        if st.button("↺  New Session", use_container_width=True, key="new_session_btn"):
            old_sid = st.session_state.session_id
            # Reset UI state immediately — no waiting for the backend call.
            reset_session()
            # Fire backend DELETE in a daemon thread so it never blocks the UI.
            threading.Thread(
                target=clear_session_sync, args=(old_sid,), daemon=True
            ).start()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
