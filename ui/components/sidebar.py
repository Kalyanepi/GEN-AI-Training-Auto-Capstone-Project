"""Sidebar — policy tier, vehicle, state, coverage selectors + session controls."""
from __future__ import annotations

import streamlit as st

from ui.utils.api_client import clear_session_sync, health_sync
from ui.utils.session_state import reset_session


VEHICLE_CATEGORIES = [
    "Economy/Compact",
    "Mid-size Sedan",
    "Full-size Sedan",
    "Compact SUV/Crossover",
    "Mid-size SUV",
    "Full-size SUV/Truck",
    "Luxury Sedan",
    "Luxury SUV",
]

US_STATES = [
    "AZ", "CA", "CO", "FL", "GA", "IL", "MI", "NC", "NJ", "NY",
    "OH", "PA", "TX", "VA", "WA",
]

COVERAGE_TYPES = [
    "(Auto-detect)", "collision", "comprehensive", "liability",
    "um_uim", "gap", "medpay", "rental", "roadside",
]

TIER_LABEL = {
    "standard": "Standard Shield",
    "premium": "Premium Guard",
    "elite": "Comprehensive Elite",
}


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## RoadGuard AI Copilot")
        st.caption("Accurate. Grounded. Cited. Never Fabricated.")
        st.divider()

        st.markdown("### Policy Tier")
        st.session_state.policy_tier = st.radio(
            label="Tier",
            options=["standard", "premium", "elite"],
            format_func=lambda t: TIER_LABEL[t],
            index=["standard", "premium", "elite"].index(st.session_state.policy_tier),
            label_visibility="collapsed",
        )

        st.markdown("### Coverage Type")
        cov = st.selectbox(
            "Coverage",
            options=COVERAGE_TYPES,
            index=0,
            label_visibility="collapsed",
        )
        st.session_state.coverage_type = None if cov == "(Auto-detect)" else cov

        st.markdown("### Vehicle Category")
        vc = st.selectbox(
            "Vehicle",
            options=["(None)"] + VEHICLE_CATEGORIES,
            index=0,
            label_visibility="collapsed",
        )
        st.session_state.vehicle_category = None if vc == "(None)" else vc

        st.markdown("### State")
        sc = st.selectbox(
            "State",
            options=["(None)"] + US_STATES,
            index=0,
            label_visibility="collapsed",
        )
        st.session_state.state_code = None if sc == "(None)" else sc

        st.divider()
        st.markdown("### Session")
        st.code(st.session_state.session_id[:8] + "...", language=None)
        if st.button("New Session", use_container_width=True):
            try:
                clear_session_sync(st.session_state.session_id)
            except Exception:
                pass
            reset_session()
            st.rerun()

        st.divider()
        with st.expander("API Health", expanded=False):
            h = health_sync()
            status_color = "green" if h.get("status") == "ok" else "orange" if h.get("status") == "degraded" else "red"
            st.markdown(f":{status_color}[{h.get('status', 'unknown')}]")
            st.json(h)
