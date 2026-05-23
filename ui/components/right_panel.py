"""Right panel — in-session 'Things I can help with' action cards."""
from __future__ import annotations

import streamlit as st

# (tabler_icon, title, subtitle, suggestions[])
CATEGORIES = [
    (
        "ti-shield-check", "Coverage & Benefits", "Understand what's covered",
        [
            "What does my Premium Guard policy cover?",
            "Are intentional acts excluded from my policy?",
            "What is the difference between collision and comprehensive coverage?",
            "Does my policy cover rental cars?",
        ],
    ),
    (
        "ti-tool", "Repair Estimates", "Get estimate ranges",
        [
            "Estimate repair cost for bumper damage on a mid-size sedan",
            "How much does windshield replacement typically cost?",
            "What is the repair cost range for airbag deployment on an SUV?",
            "Estimate rear bumper repair cost for a luxury vehicle",
        ],
    ),
    (
        "ti-calculator", "Total Loss", "Thresholds & ACV info",
        [
            "How is total loss calculated and what threshold applies to my policy?",
            "Is my car a total loss if repair is $8,500 and ACV is $12,000?",
            "What percentage of ACV triggers a total loss declaration?",
            "How does depreciation affect my total loss settlement?",
        ],
    ),
    (
        "ti-file-text", "FNOL Guidance", "Step-by-step filing help",
        [
            "How do I file a first notice of loss after an accident?",
            "What information do I need to report a claim?",
            "What happens after I file a claim?",
            "How long do I have to report an accident to my insurer?",
        ],
    ),
    (
        "ti-car", "Rental & Roadside", "Limits and coverage",
        [
            "What rental and roadside assistance benefits are included in my plan?",
            "How many days of rental coverage do I have after an accident?",
            "Does my policy cover towing and lockout service?",
            "What is my daily rental limit under Premium Guard?",
        ],
    ),
]


def handle_rp_query_param() -> None:
    if "rp" not in st.query_params:
        return
    try:
        idx = int(st.query_params["rp"])
        if 0 <= idx < len(CATEGORIES):
            # Store the category index so the chat panel renders suggestion chips.
            st.session_state.rp_category = idx
    except (ValueError, IndexError):
        pass
    st.query_params.clear()
    st.rerun()


def render_right_panel() -> None:
    st.markdown('<div class="rg-rp-header">Things I can help with</div>', unsafe_allow_html=True)
    cards = ""
    for i, (icon, title, subtitle, _suggestions) in enumerate(CATEGORIES):
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
