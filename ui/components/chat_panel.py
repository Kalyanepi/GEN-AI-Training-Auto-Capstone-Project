"""Chat panel — welcome, FAQ cards, context popup, messages, follow-up pills.

Backend wiring contract (unchanged):
  - _build_payload() sends exactly the fields api/schemas/request.py expects.
  - Response reads match api/schemas/response.py ChatResponse field names.
"""
from __future__ import annotations

import html
import random
from datetime import datetime
from typing import Optional

import streamlit as st

from ui.components.citation_card import render_citations
from ui.components.metrics_panel import render_metrics
from ui.utils.api_client import stream_chat_sync
from ui.components.right_panel import CATEGORIES as _RP_CATEGORIES

# ── FAQ pool: (tabler_icon, category_label, question, context_mode) ───
_FAQ_POOL = [
    ("ti-shield-check",   "Coverage",     "Does my Premium Guard plan cover a hit-and-run?",                    None),
    ("ti-shield-check",   "Coverage",     "Is windshield replacement covered under comprehensive?",             None),
    ("ti-tool",           "Repair",       "Estimate repair cost for rear bumper damage on a mid-size sedan",    "repair"),
    ("ti-tool",           "Repair",       "How much does front-end collision damage typically cost to repair?", "repair"),
    ("ti-scale",          "Total Loss",   "Is my car a total loss if repair is 8500 and ACV is 12000?",         "total_loss"),
    ("ti-scale",          "Total Loss",   "How is total loss threshold calculated for a 2020 Honda Accord?",    "total_loss"),
    ("ti-clipboard-list", "File a Claim", "Walk me through filing a first notice of loss",                      "state"),
    ("ti-clipboard-list", "File a Claim", "What documents do I need to start a collision claim?",               None),
    ("ti-car",            "Rental Car",   "Am I covered for a rental while my car is being repaired?",          "state"),
    ("ti-phone",          "Roadside",     "What roadside assistance benefits are included in my plan?",         None),
    ("ti-shield-half",    "UM/UIM",       "What does uninsured motorist coverage protect me from?",             None),
    ("ti-receipt",        "Deductible",   "What is my deductible for a comprehensive coverage claim?",          None),
]

# ── Follow-up pills per intent ────────────────────────────────────────
_FOLLOWUP_MAP: dict[str, list[str]] = {
    "TOTAL_LOSS": [
        "What happens next if it's declared a total loss?",
        "Can I keep my totaled vehicle?",
        "How long does total loss settlement typically take?",
    ],
    "REPAIR_ESTIMATE": [
        "Can I choose my own repair shop?",
        "Does my policy cover a rental during repairs?",
        "How do I dispute a repair estimate?",
    ],
    "COVERAGE_QA": [
        "What are my deductibles for this coverage?",
        "How do I file a claim for this?",
        "Are there any exclusions I should know about?",
    ],
    "FNOL_GUIDANCE": [
        "What documents do I need to submit?",
        "How long until I hear back after filing?",
        "Can I check my claim status online?",
    ],
    "RENTAL_LOOKUP": [
        "How many rental days am I covered for?",
        "Which rental companies are in-network?",
        "What is my daily rental reimbursement limit?",
    ],
    "ROADSIDE": [
        "How many roadside calls do I get per year?",
        "Is towing covered and up to what distance?",
        "How do I request roadside assistance?",
    ],
    "UM_UIM": [
        "What is the difference between UM and UIM?",
        "How do I file a UM/UIM claim?",
        "Does this cover hit-and-run accidents?",
    ],
}

# WHY this mapping is intentionally small: the context popup is structured
# data entry (numbers, dropdowns) — it should ONLY appear when a tool truly
# needs structured input that the user hasn't supplied. For free-text intents
# (coverage Q&A, FNOL, rental, UM/UIM) the agent can either answer directly
# from RAG or emit CLARIFICATION_NEEDED with a follow-up question — both
# better UX than a generic dropdown popup. Mapping is consulted as a hint
# only; _decide_context_mode() applies the real "missing field" check.
_INTENT_TO_CTX = {
    "TOTAL_LOSS":      "total_loss",
    "REPAIR_ESTIMATE": "repair",
}


# WHY a confidence floor: if the LLM produced a high-quality, grounded answer
# we don't need to ask for more structured input — even if some optional
# fields were missing. The popup should only nag when the answer actually
# struggled (low confidence ⇒ tools failed to find rows / produce a number).
_CONTEXT_POPUP_CONFIDENCE_FLOOR = 0.55


def _decide_context_mode(intent: Optional[str], response: dict) -> Optional[str]:
    """Return a context-popup mode only when the agent NEEDS structured input.

    Rules:
      - TOTAL_LOSS    + missing ACV or repair_cost + low confidence + NO answer → "total_loss"
      - REPAIR_ESTIMATE + missing vehicle_category + low confidence + NO answer → "repair"
      - everything else → None (no popup)

    The LLM's CLARIFICATION_NEEDED intent already handles vague queries that
    need a follow-up question, so we never auto-open the popup for those.
    """
    if not intent or intent not in _INTENT_TO_CTX:
        return None

    # If the agent already delivered a real answer, never interrupt with a form.
    if response.get("has_answer"):
        return None

    # If the agent answered confidently, don't interrupt with a form.
    confidence = response.get("confidence_score")
    if confidence is not None and confidence >= _CONTEXT_POPUP_CONFIDENCE_FLOOR:
        return None

    if intent == "TOTAL_LOSS":
        missing_acv = not st.session_state.get("acv")
        missing_repair = not st.session_state.get("repair_cost")
        if missing_acv or missing_repair:
            return "total_loss"
        return None

    if intent == "REPAIR_ESTIMATE":
        if not st.session_state.get("vehicle_category"):
            return "repair"
        return None

    return None

_GUARDRAIL_ICONS = {
    "PII_DETECTED":        "🔒",
    "PROMPT_INJECTION":    "🚫",
    "JAILBREAK_ATTEMPT":   "🚫",
    "OUT_OF_SCOPE":        "🔔",
    "LEGAL_ADVICE":        "⚖️",
    "FAULT_DETERMINATION": "⚖️",
    "MISSING_CITATION":    "📎",
    "FABRICATED_DATA":     "⚠️",
}

US_STATES = ["(None)", "AZ", "CA", "CO", "FL", "GA", "IL", "MI", "NC", "NJ", "NY", "OH", "PA", "TX", "VA", "WA"]
VEHICLE_CATEGORIES = [
    "(None)", "Economy/Compact", "Mid-size Sedan", "Full-size Sedan",
    "Compact SUV/Crossover", "Mid-size SUV", "Full-size SUV/Truck", "Luxury Sedan", "Luxury SUV",
]
COVERAGE_TYPES = [
    "(Auto-detect)", "collision", "comprehensive", "liability",
    "um_uim", "gap", "medpay", "rental", "roadside",
]


# ── Helpers ───────────────────────────────────────────────────────────

def _escape_dollars(text: str) -> str:
    return (text or "").replace("$", "\\$")


def _safe_index(options: list, value: Optional[str]) -> int:
    """Return the index of ``value`` in ``options``, defaulting to 0.

    WHY: backend param_extractor may produce values (e.g. state codes) that
    aren't in our UI whitelist. Calling list.index() with such a value raises
    ValueError and crashes the popup. Falling back to 0 ("(None)" /
    "(Auto-detect)") is the safe default.
    """
    if not value or value not in options:
        return 0
    return options.index(value)


def _now_str() -> str:
    now = datetime.now()
    h = now.strftime("%I").lstrip("0") or "12"
    return f"{h}:{now.strftime('%M %p')}"


def _get_greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


def _get_session_faqs():
    if not st.session_state.get("faq_indices"):
        st.session_state.faq_indices = random.sample(range(len(_FAQ_POOL)), min(4, len(_FAQ_POOL)))
    return [_FAQ_POOL[i] for i in st.session_state.faq_indices]


def handle_faq_query_param() -> None:
    if "faq_idx" not in st.query_params:
        return
    try:
        pool_idx = int(st.query_params["faq_idx"])
        # WHY lookup from _FAQ_POOL directly: the URL contains the actual pool
        # index (not display position), so we can retrieve the correct question
        # even if _get_session_faqs() would return a different random deck.
        if 0 <= pool_idx < len(_FAQ_POOL):
            _icon, _cat, question, ctx = _FAQ_POOL[pool_idx]
            # WHY dedupe checks: when the user clicks the browser BACK button,
            # the URL `?faq_idx=N` is restored from history → this handler
            # re-fires the same question, producing a duplicate user turn.
            # Skip if (a) a request is already pending, or (b) this exact
            # question is already the most recent user message in history.
            already_pending = bool(st.session_state.get("pending_faq"))
            msgs = st.session_state.get("messages", [])
            last_user = next(
                (m for m in reversed(msgs) if m.get("role") == "user"),
                None,
            )
            already_asked = last_user and last_user.get("content") == question
            if not already_pending and not already_asked:
                if ctx:
                    st.session_state.context_mode = ctx
                st.session_state.pending_faq = question
                # Only clear params if we actually set pending_faq.
                # If already_asked, keep URL as-is so back button works naturally.
                st.query_params.clear()
    except (ValueError, IndexError):
        pass
    # WHY no st.rerun() here: the ?faq_idx= URL navigation already triggers a
    # full Streamlit rerun. Adding another rerun causes two consecutive reloads
    # which blanks the sidebar and layout for ~4 seconds. The pending_faq set
    # above will be picked up by render_chat_panel on this same rerun cycle.


def _build_payload(message: str) -> dict:
    """Match api/schemas/request.py ChatRequest exactly."""
    payload: dict = {
        "session_id":       st.session_state.session_id,
        "message":          message,
        "policy_tier":      st.session_state.policy_tier,
        "coverage_type":    st.session_state.coverage_type,
        "vehicle_category": st.session_state.vehicle_category,
        "state_code":       st.session_state.state_code,
        "vehicle_year":     st.session_state.vehicle_year,
    }
    if st.session_state.acv:
        payload["acv"] = st.session_state.acv
    if st.session_state.repair_cost:
        payload["repair_cost"] = st.session_state.repair_cost
    return payload


# ── Context popup (above input) ───────────────────────────────────────

def _render_context_popup() -> None:
    mode = st.session_state.get("context_mode")
    if not mode:
        return

    st.markdown('<div class="rg-ctx">', unsafe_allow_html=True)

    if mode == "total_loss":
        st.markdown('<div class="rg-ctx-label">Total Loss Context</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            acv = st.number_input("ACV ($)", min_value=0, max_value=500_000,
                                  value=int(st.session_state.acv or 0), step=500, format="%d", key="ctx_acv")
            if acv > 0:
                st.session_state.acv = float(acv)
        with c2:
            rep = st.number_input("Repair ($)", min_value=0, max_value=500_000,
                                  value=int(st.session_state.repair_cost or 0), step=100, format="%d", key="ctx_repair")
            if rep > 0:
                st.session_state.repair_cost = float(rep)
        with c3:
            yr = st.number_input("Year", min_value=1990, max_value=2026,
                                 value=st.session_state.vehicle_year or 2020, step=1, format="%d", key="ctx_year")
            st.session_state.vehicle_year = int(yr)
        with c4:
            sc = st.selectbox("State", US_STATES,
                              index=_safe_index(US_STATES, st.session_state.state_code), key="ctx_state")
            st.session_state.state_code = None if sc == "(None)" else sc
        if st.session_state.acv and st.session_state.repair_cost:
            ratio = st.session_state.repair_cost / st.session_state.acv * 100
            color = "var(--green)" if ratio < 60 else ("var(--amber)" if ratio < 80 else "var(--red)")
            st.markdown(
                f"<div class='rg-stat-row' style='margin-top:6px;'>"
                f"<span class='rg-stat-lbl'>Repair ratio</span>"
                f"<span class='rg-stat-val' style='color:{color};'>{ratio:.1f}% of ACV</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    elif mode == "repair":
        st.markdown('<div class="rg-ctx-label">Repair Context</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            vc = st.selectbox("Vehicle Category", VEHICLE_CATEGORIES,
                              index=_safe_index(VEHICLE_CATEGORIES, st.session_state.vehicle_category), key="ctx_vc")
            st.session_state.vehicle_category = None if vc == "(None)" else vc
        with c2:
            yr = st.number_input("Vehicle Year", min_value=1990, max_value=2026,
                                 value=st.session_state.vehicle_year or 2020, step=1, format="%d", key="ctx_year_r")
            st.session_state.vehicle_year = int(yr)

    elif mode == "coverage":
        st.markdown('<div class="rg-ctx-label">Coverage Context</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            cur = st.session_state.coverage_type or "(Auto-detect)"
            cov = st.selectbox("Coverage Type", COVERAGE_TYPES,
                               index=COVERAGE_TYPES.index(cur) if cur in COVERAGE_TYPES else 0, key="ctx_cov")
            st.session_state.coverage_type = None if cov == "(Auto-detect)" else cov
        with c2:
            sc = st.selectbox("State", US_STATES,
                              index=_safe_index(US_STATES, st.session_state.state_code), key="ctx_state_c")
            st.session_state.state_code = None if sc == "(None)" else sc

    elif mode == "state":
        st.markdown('<div class="rg-ctx-label">Location Context</div>', unsafe_allow_html=True)
        sc = st.selectbox("State", US_STATES,
                          index=_safe_index(US_STATES, st.session_state.state_code),
                          key="ctx_state_s", label_visibility="collapsed")
        st.session_state.state_code = None if sc == "(None)" else sc

    c_d, _ = st.columns([1, 6])
    with c_d:
        if st.button("✕ Dismiss", key="ctx_dismiss"):
            st.session_state.context_mode = None
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ── Follow-up pills ───────────────────────────────────────────────────

def _render_followups(intent: Optional[str]) -> None:
    if not intent:
        return
    suggestions = _FOLLOWUP_MAP.get(intent, [])[:3]
    if not suggestions:
        return
    st.markdown('<div class="rg-followup-wrap">', unsafe_allow_html=True)
    for i, q in enumerate(suggestions):
        key_suffix = len(st.session_state.get("messages", []))
        if st.button(q, key=f"fu_{intent}_{key_suffix}_{i}"):
            st.session_state.pending_faq = q
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ── Message rendering ─────────────────────────────────────────────────

def _render_assistant_message(msg: dict, show_followups: bool = False) -> None:
    """Claude-style assistant turn: full-width left block, no avatar bubble."""
    guardrail = msg.get("guardrail_reason")
    # Wrapper for fade-in + left alignment.
    st.markdown("<div class='rg-turn rg-turn-ai'>", unsafe_allow_html=True)

    if guardrail:
        icon = _GUARDRAIL_ICONS.get(guardrail, "⚠️")
        st.markdown(
            f"<div class='rg-guardrail'>{icon} {_escape_dollars(msg['content'])}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='rg-msg-ai'>\n\n{_escape_dollars(msg['content'])}\n\n</div>",
            unsafe_allow_html=True,
        )

    if msg.get("calculation_breakdown"):
        with st.expander("📊 Calculation breakdown", expanded=True):
            st.markdown(
                f"<div class='rg-breakdown'>{msg['calculation_breakdown']}</div>",
                unsafe_allow_html=True,
            )

    render_citations(msg.get("citations") or [])

    if msg.get("disclaimer") and not guardrail:
        st.markdown(
            f"<div class='rg-disclaimer'>ℹ️ {msg['disclaimer']}</div>",
            unsafe_allow_html=True,
        )

    render_metrics(
        latency_ms=msg.get("latency_ms", 0),
        tools_used=msg.get("tools_used") or [],
        intent=msg.get("intent_detected"),
        trace_url=msg.get("trace_url"),
        confidence_score=msg.get("confidence_score"),
        guardrail_reason=guardrail,
    )

    if msg.get("time"):
        st.markdown(f"<div class='rg-ts'>{msg['time']}</div>", unsafe_allow_html=True)

    if show_followups and not guardrail:
        _render_followups(msg.get("intent_detected"))

    st.markdown("</div>", unsafe_allow_html=True)


def _render_user_message(msg: dict) -> None:
    """Claude-style user turn: right-aligned bubble, no avatar."""
    # WHY html.escape: raw HTML — markdown's backslash escape would show literal "\$".
    st.markdown(
        f"<div class='rg-turn rg-turn-user'>"
        f"  <div class='rg-msg-user'>{html.escape(msg['content'] or '')}</div>"
        + (f"  <div class='rg-ts rg-ts-user'>{msg['time']}</div>" if msg.get("time") else "")
        + "</div>",
        unsafe_allow_html=True,
    )


# ── Thinking placeholder ──────────────────────────────────────────────

_THINKING_HTML = """
<div class='rg-turn rg-turn-ai'>
  <div class='rg-thinking'>
    <span class='rg-thinking-dots'>
      <span></span><span></span><span></span>
    </span>
    <span class='rg-thinking-text'>{label}</span>
  </div>
</div>
"""


def _render_thinking(placeholder, label: str = "Thinking…") -> None:
    placeholder.markdown(_THINKING_HTML.format(label=label), unsafe_allow_html=True)


# ── Send message ──────────────────────────────────────────────────────

def _send_message(user_input: str) -> None:
    ts = _now_str()
    st.session_state.messages.append({"role": "user", "content": user_input, "time": ts})
    _render_user_message({"content": user_input, "time": ts})

    # Thinking bubble shown while tools/guardrails run (before first token).
    thinking_slot = st.empty()
    _render_thinking(thinking_slot, "Thinking…")

    # State collected from SSE events.
    meta: dict = {}
    full_text: str = ""
    done_data: dict = {}
    guardrail_data: dict = {}
    stream_slot = None  # placeholder for the streaming text area
    _token_buf: int = 0
    _RENDER_EVERY = 6  # re-render every N tokens to reduce Streamlit overhead

    try:
        for event_type, data in stream_chat_sync(_build_payload(user_input)):
            if event_type == "meta":
                meta = data
                # Swap thinking bubble for streaming text area.
                thinking_slot.empty()
                stream_slot = st.empty()

            elif event_type == "token":
                token = data if isinstance(data, str) else data
                full_text += token
                _token_buf += 1
                if stream_slot is not None and _token_buf >= _RENDER_EVERY:
                    _token_buf = 0
                    stream_slot.markdown(
                        f"<div class='rg-turn rg-turn-ai'>"
                        f"<div class='rg-msg-ai'>\n\n{_escape_dollars(full_text)}\n\n</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

            elif event_type == "guardrail":
                guardrail_data = data if isinstance(data, dict) else {}

            elif event_type == "done":
                done_data = data if isinstance(data, dict) else {}

    except Exception as e:
        thinking_slot.empty()
        err = f"Connection error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": err, "time": _now_str()})
        _render_assistant_message(
            {"role": "assistant", "content": err, "time": _now_str()},
            show_followups=False,
        )
        return

    # Clear streaming slot — we'll re-render cleanly via _render_assistant_message.
    thinking_slot.empty()
    if stream_slot:
        stream_slot.empty()

    # Handle guardrail-blocked answer.
    guardrail_reason = guardrail_data.get("reason") or meta.get("guardrail_reason")
    if done_data.get("guardrail_triggered") or guardrail_reason:
        final_content = guardrail_data.get("message") or full_text
    else:
        final_content = full_text or "(no answer)"

    intent = meta.get("intent")
    citations = meta.get("citations") or []
    tools_used = meta.get("tools") or []
    trace_id = meta.get("trace_id")
    from observability.langsmith_tracer import build_trace_url
    trace_url = build_trace_url(trace_id) if trace_id else None

    assistant_msg = {
        "role":                  "assistant",
        "content":               final_content,
        "citations":             citations,
        "latency_ms":            done_data.get("latency_ms", 0),
        "tools_used":            tools_used,
        "intent_detected":       intent,
        "trace_url":             trace_url,
        "guardrail_reason":      guardrail_reason,
        "disclaimer":            done_data.get("disclaimer"),
        "calculation_breakdown": done_data.get("calculation_breakdown"),
        "confidence_score":      done_data.get("confidence_score"),
        "time":                  _now_str(),
    }

    st.session_state.context_mode = _decide_context_mode(intent, {
        "intent_detected": intent,
        "confidence_score": done_data.get("confidence_score"),
        "guardrail_triggered": bool(guardrail_reason),
        # has_answer: True when the agent produced a real response — suppress popup.
        "has_answer": bool(final_content and len(final_content.strip()) > 40),
    })
    st.session_state.messages.append(assistant_msg)
    _render_assistant_message(assistant_msg, show_followups=True)


# ── Welcome screen ────────────────────────────────────────────────────

def _render_welcome() -> None:
    greeting = _get_greeting()

    st.markdown(
        f"""<div class="rg-welcome">
          <div class="rg-welcome-badge"><i class="ti ti-sparkles"></i></div>
          <div class="rg-welcome-title">{greeting}</div>
          <div class="rg-welcome-sub">
            I am Auto Insurance AI Copilot, how can I help you today?
          </div>
          <div class="rg-feat-row">
            <span class="rg-feat-tag"><i class="ti ti-circle-check"></i> Accurate</span>
            <span class="rg-feat-tag"><i class="ti ti-anchor"></i> Grounded</span>
            <span class="rg-feat-tag"><i class="ti ti-quote"></i> Cited</span>
            <span class="rg-feat-tag"><i class="ti ti-shield-lock"></i> Never Fabricated</span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rg-faq-label">Try asking</div>', unsafe_allow_html=True)
    faqs = _get_session_faqs()

    faq_left, faq_right = "", ""
    for disp_i, (icon, category, question, _ctx) in enumerate(faqs):
        # WHY actual_pool_idx: We store the actual index into _FAQ_POOL in the URL,
        # not the display position (0-3). This prevents question mismatch when
        # the page reloads and _get_session_faqs() returns a different random deck.
        actual_pool_idx = st.session_state.faq_indices[disp_i]
        card = (
            f'<a href="?faq_idx={actual_pool_idx}" target="_self" class="rg-faq-card">'
            f'  <span class="rg-faq-icon-wrap"><i class="ti {icon}"></i></span>'
            f'  <span class="rg-faq-body">'
            f'    <span class="rg-faq-cat">{category}</span>'
            f'    <span class="rg-faq-q">{question}</span>'
            f'  </span>'
            f'</a>'
        )
        if disp_i % 2 == 0:
            faq_left += card
        else:
            faq_right += card

    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.markdown(faq_left, unsafe_allow_html=True)
    with col2:
        st.markdown(faq_right, unsafe_allow_html=True)


# ── Main entry points ─────────────────────────────────────────────────

def render_chat_panel() -> None:
    """Renders chat content only (no input bar)."""
    # Immediately after New Session — suppress the welcome flash by rendering
    # nothing on this first rerun, then rerun immediately so the welcome
    # screen renders cleanly on the very next cycle.
    if st.session_state.pop("_resetting", False):
        st.rerun()

    msgs = st.session_state.messages
    pending = st.session_state.get("pending_faq")

    # Right-panel card was clicked — show category suggestion chips.
    rp_cat = st.session_state.get("rp_category")
    if rp_cat is not None and not pending:
        _icon, title, _sub, suggestions = _RP_CATEGORIES[rp_cat]
        st.markdown(
            f"<div class='rg-sugg-header'>"
            f"<i class='ti {_icon}'></i> {title}"
            f"</div>",
            unsafe_allow_html=True,
        )
        for q in suggestions:
            if st.button(q, key=f"rp_sugg_{rp_cat}_{q[:30]}", use_container_width=True):
                del st.session_state["rp_category"]
                st.session_state.pending_faq = q
                st.rerun()
        if st.button("✕ Dismiss", key="rp_sugg_dismiss"):
            del st.session_state["rp_category"]
            st.rerun()
        return

    # Welcome screen only when truly idle.
    if not msgs and not pending:
        _render_welcome()
        return

    # WHY enumerate with index check: for very long conversations (50+ turns),
    # we could implement virtual scrolling here. Currently we render all
    # messages which is fine for <30 turns but may lag with 100+.
    # If performance degrades, add: if idx < max_visible: render, else: break.
    for idx, msg in enumerate(msgs):
        # Don't show follow-up pills on the last historical assistant turn
        # if we are about to render a fresh turn underneath it.
        is_last_historical = (idx == len(msgs) - 1) and not pending
        if msg["role"] == "user":
            _render_user_message(msg)
        else:
            _render_assistant_message(msg, show_followups=is_last_historical)

    # Then process any pending message — _send_message renders the new
    # user turn + thinking placeholder + final assistant turn inline.
    if pending:
        question = st.session_state.pop("pending_faq")
        _send_message(question)


def render_chat_input() -> None:
    """Renders the chat input bar. Must be called OUTSIDE st.columns."""
    _render_context_popup()

    user_input = st.chat_input("Ask anything about your policy...")
    if user_input:
        # WHY rerun-via-pending_faq: render_chat_input runs AFTER render_chat_panel,
        # which means a fresh submission would otherwise render the welcome screen
        # above the new chat turn. Queueing the message and rerunning ensures
        # render_chat_panel sees the pending message and skips the welcome entirely.
        st.session_state.pending_faq = user_input
        st.rerun()
    st.markdown(
        "<div class='rg-input-note'>Auto Insurance AI Copilot can make mistakes. Verify important info.</div>",
        unsafe_allow_html=True,
    )
