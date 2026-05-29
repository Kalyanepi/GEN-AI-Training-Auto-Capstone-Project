"""Streamlit session state initialization and reset."""
from __future__ import annotations

import uuid
import streamlit as st


def init_session_state() -> None:
    """Set up all keys on first page load."""
    defaults = {
        "session_id":       str(uuid.uuid4()),
        "messages":         [],
        "policy_tier":      "premium",
        "vehicle_category": None,
        "state_code":       None,
        "coverage_type":    None,
        "vehicle_year":     None,
        "acv":              None,
        "repair_cost":      None,
        # UI state
        "context_mode":     None,   # drives the dynamic context popup
        "pending_faq":      None,   # FAQ question queued for send
        "faq_indices":      None,   # randomly sampled FAQ indices for this session
        "ui_theme":         "light",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_session() -> None:
    """Generate a new session and clear chat + context state."""
    st.session_state.session_id       = str(uuid.uuid4())
    st.session_state.messages         = []
    st.session_state.vehicle_year     = None
    st.session_state.acv              = None
    st.session_state.repair_cost      = None
    st.session_state.vehicle_category = None
    st.session_state.state_code       = None
    st.session_state.coverage_type    = None
    st.session_state.context_mode     = None
    st.session_state.pending_faq      = None
    st.session_state.faq_indices      = None   # forces new random FAQ set
    st.session_state["_resetting"]    = True   # suppresses welcome flash on rerun
