"""Streamlit session state initialization."""
from __future__ import annotations

import uuid

import streamlit as st


def init_session_state() -> None:
    """Set up keys on first page load."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{role, content, citations, latency_ms, tools_used, breakdown}]
    if "policy_tier" not in st.session_state:
        st.session_state.policy_tier = "premium"
    if "vehicle_category" not in st.session_state:
        st.session_state.vehicle_category = None
    if "state_code" not in st.session_state:
        st.session_state.state_code = None
    if "coverage_type" not in st.session_state:
        st.session_state.coverage_type = None


def reset_session() -> None:
    """Generate a new session id and clear chat — used by the New Session button."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
