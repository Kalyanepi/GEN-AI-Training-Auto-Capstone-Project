"""Streamlit entrypoint — RoadGuard AI Copilot."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ui.components.chat_panel import render_chat_panel
from ui.components.sidebar import render_sidebar
from ui.utils.session_state import init_session_state

# ---------------------------------------------------------------------------
# Custom CSS — injected once at startup
# ---------------------------------------------------------------------------
_CSS = """
<style>
/* ── Brand palette ──────────────────────────────────────────────── */
:root {
    --rg-navy:   #1a2744;
    --rg-red:    #e74c3c;
    --rg-green:  #27ae60;
    --rg-orange: #e67e22;
    --rg-gray:   #f4f6f9;
    --rg-border: #dce3ed;
    --rg-text:   #2c3e50;
}

/* ── Global typography ──────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    color: var(--rg-text);
}

/* ── Header ─────────────────────────────────────────────────────── */
.rg-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 0 6px 0;
    border-bottom: 2px solid var(--rg-navy);
    margin-bottom: 4px;
}
.rg-header h1 {
    margin: 0;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--rg-navy);
}
.rg-tagline {
    font-size: 0.82rem;
    color: #6b7a99;
    margin-top: 2px;
    letter-spacing: 0.02em;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--rg-navy) !important;
}
[data-testid="stSidebar"] * {
    color: #e8edf5 !important;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #c5cfe0 !important;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stSidebar"] hr {
    border-color: #2e4070 !important;
}
[data-testid="stSidebar"] .stButton button {
    background: #e74c3c !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #c0392b !important;
}

/* ── Chat bubbles ────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 10px;
    padding: 2px 4px;
}

/* ── Citation cards ──────────────────────────────────────────────── */
.rg-citation-pdf {
    border-left: 3px solid #2980b9;
    padding-left: 8px;
    background: #eaf3fb;
    border-radius: 4px;
}
.rg-citation-csv {
    border-left: 3px solid #27ae60;
    padding-left: 8px;
    background: #eafaf1;
    border-radius: 4px;
}

/* ── Metrics row ─────────────────────────────────────────────────── */
.rg-metrics {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 6px;
    font-size: 0.78rem;
    color: #5d6d8a;
}
.rg-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.75rem;
}
.rg-pill-green  { background: #d5f5e3; color: #1a7a40; }
.rg-pill-orange { background: #fdebd0; color: #a04000; }
.rg-pill-red    { background: #fadbd8; color: #922b21; }
.rg-pill-blue   { background: #d6eaf8; color: #1a5276; }
.rg-pill-gray   { background: #eaecee; color: #4d5656; }

/* ── Guardrail alert ─────────────────────────────────────────────── */
.rg-guardrail {
    background: #fdf3cd;
    border: 1px solid #f0c040;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.88rem;
}

/* ── Calculation breakdown ───────────────────────────────────────── */
.rg-breakdown {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 10px 14px;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 0.80rem;
    line-height: 1.6;
}

/* ── Tier badge ──────────────────────────────────────────────────── */
.rg-tier-standard    { color: #7f8c8d; font-weight: 600; }
.rg-tier-premium     { color: #2980b9; font-weight: 600; }
.rg-tier-elite       { color: #8e44ad; font-weight: 600; }

/* ── Disclaimer ──────────────────────────────────────────────────── */
.rg-disclaimer {
    font-size: 0.75rem;
    color: #7f8c8d;
    font-style: italic;
    margin-top: 4px;
}
</style>
"""


def main() -> None:
    st.set_page_config(
        page_title="RoadGuard AI Copilot",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject CSS once.
    st.markdown(_CSS, unsafe_allow_html=True)

    init_session_state()
    render_sidebar()

    # Header.
    st.markdown(
        """
        <div class="rg-header">
            <span style="font-size:2rem;">🛡️</span>
            <div>
                <h1>RoadGuard AI Copilot</h1>
                <div class="rg-tagline">Accurate &nbsp;·&nbsp; Grounded &nbsp;·&nbsp; Cited &nbsp;·&nbsp; Never Fabricated</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Hi! I'm **RoadGuard AI Copilot**. Set your policy tier and state in the sidebar, then ask me about:\n\n"
                "- 📋 **Coverage questions** — what's covered, exclusions, deductibles\n"
                "- 🔧 **Repair estimates** — damage type + vehicle category cost ranges\n"
                "- ⚖️ **Total loss** — state-specific threshold + ACV calculation\n"
                "- 📝 **FNOL guidance** — step-by-step claim filing instructions\n"
                "- 🚗 **Rental & Roadside** — tier-aware daily limits and towing coverage\n\n"
                "_For total loss queries, enter your ACV and repair cost in the sidebar for best results._"
            )

    render_chat_panel()


if __name__ == "__main__":
    main()
