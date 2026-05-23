"""Right panel — in-session 'Things I can help with' action cards."""
from __future__ import annotations

import streamlit as st

# (tabler_icon, title, subtitle, context_mode, question)
CATEGORIES = [
    ("ti-shield-check",   "Coverage & Benefits", "Understand what's covered",    "coverage",   "What does my Premium Guard policy cover?"),
    ("ti-tool",           "Repair Estimates",    "Get estimate ranges",          "repair",     "Estimate repair cost for bumper damage on a mid-size sedan"),
    ("ti-calculator",     "Total Loss",          "Thresholds & ACV info",        "total_loss", "How is total loss calculated and what threshold applies to my policy?"),
    ("ti-file-text",      "FNOL Guidance",       "Step-by-step filing help",     "state",      "How do I file a first notice of loss after an accident?"),
    ("ti-car",            "Rental & Roadside",   "Limits and coverage",          None,         "What rental and roadside assistance benefits are included in my plan?"),
]


def handle_rp_query_param() -> None:
    if "rp" not in st.query_params:
        return
    try:
        idx = int(st.query_params["rp"])
        if 0 <= idx < len(CATEGORIES):
            _icon, _title, _sub, ctx, question = CATEGORIES[idx]
            if ctx:
                st.session_state.context_mode = ctx
            st.session_state.pending_faq = question
    except (ValueError, IndexError):
        pass
    st.query_params.clear()
    st.rerun()


def render_right_panel() -> None:
    st.markdown('<div class="rg-rp-header">Things I can help with</div>', unsafe_allow_html=True)
    cards = ""
    for i, (icon, title, subtitle, _ctx, _q) in enumerate(CATEGORIES):
        cards += (
            f'<a href="?rp={i}" target="_self" class="rg-rp-card">'
            f'  <span class="rg-rp-icon"><i class="ti {icon}"></i></span>'
            f'  <span class="rg-rp-body">'
            f'    <span class="rg-rp-title">{title}</span>'
            f'    <span class="rg-rp-sub">{subtitle}</span>'
            f'  </span>'
            f'  <i class="ti ti-chevron-right rg-rp-chev"></i>'
            f'</a>'
        )
    st.markdown(cards, unsafe_allow_html=True)
