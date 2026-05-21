"""Streamlit entrypoint."""
from __future__ import annotations

# WHY this sys.path tweak: streamlit runs this file as a script, not as a
# module, so the project root isn't on sys.path and `from ui...` imports fail.
# Prepending the parent dir fixes it without requiring PYTHONPATH env vars.
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ui.components.chat_panel import render_chat_panel
from ui.components.sidebar import render_sidebar
from ui.utils.session_state import init_session_state


def main() -> None:
    st.set_page_config(
        page_title="RoadGuard AI Copilot",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    render_sidebar()

    st.markdown("# 🚗 RoadGuard AI Copilot")
    st.caption("Auto Insurance — Accurate. Grounded. Cited. Never Fabricated.")

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Hi! I'm RoadGuard AI Copilot. I can help with:\n\n"
                "- **Coverage questions** — what's covered under your tier\n"
                "- **Repair estimates** — damage type + vehicle category lookups\n"
                "- **Total loss** — state-specific threshold + ACV calculation\n"
                "- **FNOL guidance** — step-by-step claim filing\n"
                "- **Rental / Roadside** — tier-aware benefits\n\n"
                "Set your policy tier and state in the sidebar, then ask away."
            )

    render_chat_panel()


if __name__ == "__main__":
    main()
