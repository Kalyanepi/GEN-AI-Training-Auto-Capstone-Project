"""Sidebar — policy tier, vehicle, state, ACV/repair selectors + session controls."""
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

# Friendly display labels for each tier.
TIER_LABEL = {
    "standard": "Standard Shield",
    "premium":  "Premium Guard",
    "elite":    "Comprehensive Elite",
}

# Deductible summary shown under each tier — helps users pick quickly.
TIER_DEDUCTIBLE_HINT = {
    "standard": "Collision $1,000 · Comp $500",
    "premium":  "Collision $500 · Comp $250",
    "elite":    "Collision $250 · Comp $0",
}

TIER_COLORS = {
    "standard": "⬜",
    "premium":  "🔵",
    "elite":    "🟣",
}


def render_sidebar() -> None:
    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────
        st.markdown(
            "<div style='text-align:center; padding:8px 0 4px;'>"
            "<span style='font-size:2.2rem;'>🛡️</span><br>"
            "<span style='font-size:1.05rem; font-weight:700; letter-spacing:0.01em;'>"
            "RoadGuard AI</span><br>"
            "<span style='font-size:0.72rem; opacity:0.65;'>Auto Insurance Copilot</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Policy Tier ───────────────────────────────────────────
        st.markdown("**POLICY TIER**")
        tier_options = ["standard", "premium", "elite"]
        tier_index = tier_options.index(st.session_state.policy_tier)
        selected_tier = st.radio(
            label="tier",
            options=tier_options,
            format_func=lambda t: f"{TIER_COLORS[t]} {TIER_LABEL[t]}",
            index=tier_index,
            label_visibility="collapsed",
        )
        st.session_state.policy_tier = selected_tier
        st.caption(TIER_DEDUCTIBLE_HINT[selected_tier])

        st.divider()

        # ── Coverage Type ─────────────────────────────────────────
        st.markdown("**COVERAGE TYPE**")
        cov = st.selectbox(
            "coverage",
            options=COVERAGE_TYPES,
            index=0,
            label_visibility="collapsed",
        )
        st.session_state.coverage_type = None if cov == "(Auto-detect)" else cov

        # ── Vehicle Category ──────────────────────────────────────
        st.markdown("**VEHICLE CATEGORY**")
        vc_options = ["(None)"] + VEHICLE_CATEGORIES
        current_vc = st.session_state.vehicle_category or "(None)"
        vc_index = vc_options.index(current_vc) if current_vc in vc_options else 0
        vc = st.selectbox(
            "vehicle",
            options=vc_options,
            index=vc_index,
            label_visibility="collapsed",
        )
        st.session_state.vehicle_category = None if vc == "(None)" else vc

        # ── Vehicle Year ──────────────────────────────────────────
        st.markdown("**VEHICLE YEAR**")
        vehicle_year_val = st.number_input(
            "year",
            min_value=1990,
            max_value=2026,
            value=st.session_state.vehicle_year or 2020,
            step=1,
            format="%d",
            label_visibility="collapsed",
            help="Used to determine the age bucket for total loss threshold.",
        )
        st.session_state.vehicle_year = int(vehicle_year_val)

        # ── State ─────────────────────────────────────────────────
        st.markdown("**STATE**")
        sc_options = ["(None)"] + US_STATES
        current_sc = st.session_state.state_code or "(None)"
        sc_index = sc_options.index(current_sc) if current_sc in sc_options else 0
        sc = st.selectbox(
            "state",
            options=sc_options,
            index=sc_index,
            label_visibility="collapsed",
        )
        st.session_state.state_code = None if sc == "(None)" else sc

        st.divider()

        # ── Total Loss Inputs ─────────────────────────────────────
        st.markdown("**TOTAL LOSS INPUTS**")
        st.caption("Pre-fill for faster total loss calculations.")

        acv_val = st.number_input(
            "Actual Cash Value (ACV) $",
            min_value=0,
            max_value=500_000,
            value=int(st.session_state.acv or 0),
            step=500,
            format="%d",
            help="Vehicle's current market value before the accident.",
        )
        st.session_state.acv = float(acv_val) if acv_val > 0 else None

        repair_val = st.number_input(
            "Repair Cost Estimate $",
            min_value=0,
            max_value=500_000,
            value=int(st.session_state.repair_cost or 0),
            step=100,
            format="%d",
            help="Shop estimate to repair the vehicle.",
        )
        st.session_state.repair_cost = float(repair_val) if repair_val > 0 else None

        if st.session_state.acv and st.session_state.repair_cost:
            ratio = st.session_state.repair_cost / st.session_state.acv * 100
            color = "🟢" if ratio < 60 else ("🟡" if ratio < 80 else "🔴")
            st.caption(f"{color} Repair ratio: **{ratio:.1f}%** of ACV")

        st.divider()

        # ── Session ───────────────────────────────────────────────
        st.markdown("**SESSION**")
        st.code(st.session_state.session_id[:12] + "...", language=None)
        if st.button("🔄 New Session", use_container_width=True):
            try:
                clear_session_sync(st.session_state.session_id)
            except Exception:
                pass
            reset_session()
            st.rerun()

        # ── API Health (collapsed by default) ─────────────────────
        with st.expander("API Health", expanded=False):
            h = health_sync()
            status = h.get("status", "unknown")
            icon = "🟢" if status == "ok" else ("🟡" if status == "degraded" else "🔴")
            st.markdown(f"{icon} **{status}**")
            if status != "ok":
                st.json(h)
