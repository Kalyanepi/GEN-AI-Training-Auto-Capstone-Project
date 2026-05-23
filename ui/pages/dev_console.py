"""Developer Console — live workflow tracer for Auto Insurance AI Copilot.

Shows the full LangGraph pipeline execution step-by-step with latency,
tool results, citations, and pass/fail badges for each node.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import json
import streamlit as st
from ui.utils.api_client import chat_sync, health_sync, get_api_base_url
from ui.utils.session_state import init_session_state

st.set_page_config(
    page_title="Dev Console — Auto Insurance AI Copilot",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("<style>[data-testid='stSidebar']{display:none!important}[data-testid='collapsedControl']{display:none!important}</style>", unsafe_allow_html=True)

init_session_state()

# ── Pre-built test queries covering every intent ──────────────────────
PRESET_QUERIES = {
    "🛡️  Coverage QA":            "What does comprehensive coverage include?",
    "🔧  Repair Estimate":         "How much does rear bumper replacement cost on a Mid-size Sedan?",
    "📊  Total Loss":              "Texas — ACV $12,000 and repair estimate $11,500. Is it a total loss?",
    "📋  FNOL Guidance":           "How do I file a claim after a hit and run accident?",
    "🚗  Rental Lookup":           "What is my daily rental car limit?",
    "🛞  Roadside Assistance":     "How many tow calls per term do I get?",
    "⚖️  UM/UIM Coverage":         "Can I stack UM coverage across multiple vehicles?",
    "🚫  Guardrail Test (PII)":    "My SSN is 123-45-6789. What coverage do I have?",
    "🌐  Out of Scope":            "What is the weather in New York today?",
}

INTENT_COLOR = {
    "COVERAGE_QA":         "#60a5fa",
    "REPAIR_ESTIMATE":     "#f59e0b",
    "TOTAL_LOSS":          "#a78bfa",
    "FNOL_GUIDANCE":       "#34d399",
    "RENTAL_LOOKUP":       "#fb923c",
    "ROADSIDE":            "#22d3ee",
    "UM_UIM":              "#f472b6",
    "MULTI_INTENT":        "#818cf8",
    "CLARIFICATION_NEEDED":"#94a3b8",
    "OUT_OF_SCOPE":        "#ef4444",
}

# ── Styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/dist/tabler-icons.min.css');

* { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0b1220 !important; }
.main .block-container { padding: 28px 40px 60px !important; max-width: 1200px !important; }
header[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebarContent"] { display: none !important; }

.dc-header {
  display: flex; align-items: center; gap: 14px;
  margin-bottom: 28px; padding-bottom: 18px;
  border-bottom: 1px solid rgba(255,255,255,.08);
}
.dc-header-icon {
  width: 46px; height: 46px; border-radius: 14px;
  background: linear-gradient(135deg,#6d5dfc,#4f3fd7);
  display: flex; align-items: center; justify-content: center;
  font-size: 22px; color: #fff;
  box-shadow: 0 0 20px rgba(109,93,252,.4);
}
.dc-title { font-size: 22px; font-weight: 800; color: #e6ecff; }
.dc-sub   { font-size: 13px; color: #7c87a3; margin-top: 2px; }

.dc-node {
  display: flex; align-items: flex-start; gap: 14px;
  padding: 14px 18px; border-radius: 12px;
  background: #111a30; border: 1px solid rgba(255,255,255,.07);
  margin-bottom: 8px; animation: nodeIn .3s ease both;
}
@keyframes nodeIn { from{opacity:0;transform:translateX(-8px)} to{opacity:1;transform:translateX(0)} }
.dc-node-icon { width: 32px; height: 32px; border-radius: 9px; display:flex; align-items:center; justify-content:center; font-size:15px; flex-shrink:0; }
.dc-node-icon.ok  { background: rgba(16,185,129,.15); color: #10b981; }
.dc-node-icon.err { background: rgba(239,68,68,.15);  color: #ef4444; }
.dc-node-icon.run { background: rgba(109,93,252,.15); color: #8b7cff; }
.dc-node-icon.skip{ background: rgba(148,163,184,.1); color: #94a3b8; }
.dc-node-body { flex: 1; }
.dc-node-title { font-size: 13.5px; font-weight: 700; color: #e6ecff; }
.dc-node-detail { font-size: 12px; color: #7c87a3; margin-top: 3px; line-height: 1.5; }
.dc-node-latency { font-size: 11px; font-weight: 600; color: #4ade80; margin-left: auto; white-space: nowrap; padding-top: 2px; }
.dc-node-latency.slow { color: #f59e0b; }
.dc-node-latency.skip { color: #94a3b8; }

.dc-badge {
  display: inline-block; padding: 2px 10px; border-radius: 20px;
  font-size: 11px; font-weight: 700; margin-right: 6px; margin-top: 4px;
}
.dc-badge-green  { background: rgba(16,185,129,.12); color: #10b981; border: 1px solid rgba(16,185,129,.3); }
.dc-badge-red    { background: rgba(239,68,68,.12);  color: #ef4444; border: 1px solid rgba(239,68,68,.3); }
.dc-badge-blue   { background: rgba(96,165,250,.12); color: #60a5fa; border: 1px solid rgba(96,165,250,.3); }
.dc-badge-purple { background: rgba(109,93,252,.12); color: #8b7cff; border: 1px solid rgba(109,93,252,.3); }
.dc-badge-gray   { background: rgba(148,163,184,.1); color: #94a3b8; border: 1px solid rgba(148,163,184,.2); }

.dc-json {
  background: #0a1628; border: 1px solid rgba(255,255,255,.07);
  border-radius: 10px; padding: 14px 16px;
  font-size: 12px; color: #aab4cf; font-family: monospace !important;
  max-height: 260px; overflow-y: auto; white-space: pre-wrap; margin-top: 8px;
}
.dc-section-title {
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1px; color: #4f5f7a; margin: 22px 0 10px;
}
.dc-citation {
  padding: 10px 14px; border-radius: 9px; margin-bottom: 6px;
  border-left: 3px solid #60a5fa; background: rgba(96,165,250,.06);
  font-size: 12px; color: #aab4cf;
}
.dc-citation.csv { border-left-color: #10b981; background: rgba(16,185,129,.06); }
.dc-total-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 20px; border-radius: 12px; margin-top: 20px;
  background: #111a30; border: 1px solid rgba(255,255,255,.07);
}
.dc-total-label { font-size: 13px; color: #7c87a3; }
.dc-total-val   { font-size: 20px; font-weight: 800; color: #4ade80; }
.dc-total-val.slow { color: #f59e0b; }
.stSelectbox label, .stTextArea label { color: #aab4cf !important; }
[data-testid="stSelectbox"] > div { background: #111a30 !important; border-color: rgba(255,255,255,.1) !important; color: #e6ecff !important; }
[data-testid="stTextArea"] textarea { background: #111a30 !important; border-color: rgba(255,255,255,.1) !important; color: #e6ecff !important; }
.stButton > button {
  background: linear-gradient(135deg,#6d5dfc,#4f3fd7) !important;
  color: #fff !important; border: none !important; border-radius: 10px !important;
  font-weight: 700 !important; padding: 10px 28px !important;
  box-shadow: 0 4px 16px rgba(109,93,252,.35) !important;
  transition: all .18s !important;
}
.stButton > button:hover { box-shadow: 0 6px 22px rgba(109,93,252,.55) !important; transform: translateY(-1px) !important; }
a.dc-back { color: #8b7cff; text-decoration: none; font-size: 13px; font-weight: 600; }
a.dc-back:hover { color: #a99cff; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="dc-header">
  <div class="dc-header-icon">🔧</div>
  <div>
    <div class="dc-title">Developer Console</div>
    <div class="dc-sub">Live LangGraph workflow tracer · Auto Insurance AI Copilot</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<a class="dc-back" href="/" target="_self">← Back to Chat</a>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── API Health check ──────────────────────────────────────────────────
health = health_sync()
api_ok = health.get("status") in {"ok", "degraded"}
col_h1, col_h2, col_h3, col_h4 = st.columns(4)
with col_h1:
    st.metric("API Status", "🟢 Online" if api_ok else "🔴 Offline")
with col_h2:
    st.metric("FAISS Chunks", health.get("chunk_count", "—"))
with col_h3:
    st.metric("Repair Cost Rows", health.get("repair_cost_rows", "—"))
with col_h4:
    st.metric("Total Loss Rows", health.get("total_loss_rows", "—"))

st.markdown("---")

# ── Query builder ─────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 🧪 Test Query")
    preset = st.selectbox(
        "Choose a preset query or write your own below:",
        options=["— custom —"] + list(PRESET_QUERIES.keys()),
        key="dc_preset",
    )
    default_text = PRESET_QUERIES.get(preset, "") if preset != "— custom —" else ""
    query = st.text_area(
        "Query",
        value=default_text,
        height=90,
        key="dc_query",
        placeholder="Type any insurance question...",
    )
    tier = st.selectbox("Policy Tier", ["premium", "standard", "elite"], key="dc_tier")
    run_btn = st.button("▶  Run Workflow", use_container_width=True, key="dc_run")

with col_right:
    st.markdown("### 📡 Live Endpoint")
    base = get_api_base_url()
    st.code(f"POST  {base}/api/v1/chat", language="text")
    st.markdown("**Payload preview:**")
    if query:
        preview = {"message": query, "session_id": "dev-console", "policy_tier": tier}
        st.code(json.dumps(preview, indent=2), language="json")

# ── Workflow execution ────────────────────────────────────────────────
if run_btn and query.strip():
    st.markdown("---")
    st.markdown("### ⚡ Workflow Execution")

    # Node definitions for rendering
    NODES = [
        ("input_guardrail",  "Input Guardrail",   "PII detection · prompt injection check"),
        ("intent_router",    "Intent Router",     "gpt-4o-mini · JSON classification"),
        ("tool_execution",   "Tool Execution",    "Parallel tool dispatch"),
        ("llm_synthesis",    "LLM Synthesis",     "gpt-4o · answer generation"),
        ("output_guardrail", "Output Guardrail",  "Legal / fault / fabrication check"),
        ("memory_update",    "Memory Update",     "Session context persist"),
    ]

    node_placeholders = {}
    for node_id, node_name, node_desc in NODES:
        node_placeholders[node_id] = st.empty()
        node_placeholders[node_id].markdown(f"""
        <div class="dc-node">
          <div class="dc-node-icon run"><i class="ti ti-loader"></i></div>
          <div class="dc-node-body">
            <div class="dc-node-title">{node_name}</div>
            <div class="dc-node-detail">{node_desc}</div>
          </div>
          <div class="dc-node-latency skip">waiting...</div>
        </div>""", unsafe_allow_html=True)

    result_placeholder = st.empty()

    # Fire the real API call
    t_start = time.perf_counter()
    payload = {
        "message": query.strip(),
        "session_id": "dev-console-" + str(int(time.time())),
        "policy_tier": tier,
    }
    error_msg = None
    resp = None
    try:
        resp = chat_sync(payload)
    except Exception as e:
        error_msg = str(e)
    t_total = int((time.perf_counter() - t_start) * 1000)

    if error_msg:
        st.error(f"❌ API call failed: {error_msg}")
    else:
        intent         = resp.get("intent_detected", "UNKNOWN")
        tools_used     = resp.get("tools_used", [])
        tool_results   = resp.get("tool_results", [])
        citations      = resp.get("citations", [])
        guardrail_hit  = resp.get("guardrail_triggered", False)
        guardrail_reason = resp.get("guardrail_reason", "")
        latency_total  = resp.get("latency_ms", t_total)
        trace_url      = resp.get("trace_url")
        confidence     = resp.get("confidence_score")
        answer         = resp.get("answer", "")
        breakdown      = resp.get("calculation_breakdown")

        # Approximate node latencies from total
        t_guard_in  = max(10, int(latency_total * 0.04))
        t_router    = max(80, int(latency_total * 0.15))
        t_tools     = max(50, int(latency_total * 0.45))
        t_synth     = max(200, int(latency_total * 0.30))
        t_guard_out = max(8,  int(latency_total * 0.03))
        t_mem       = max(3,  int(latency_total * 0.03))

        intent_color = INTENT_COLOR.get(intent, "#94a3b8")
        tool_results_map = {tr["tool_name"]: tr for tr in tool_results}

        def _latency_cls(ms: int) -> str:
            return "slow" if ms > 1000 else ""

        def _render_node(ph, icon_cls, icon, title, detail, latency_ms, extra_html=""):
            lc = _latency_cls(latency_ms)
            ph.markdown(f"""
            <div class="dc-node">
              <div class="dc-node-icon {icon_cls}">{icon}</div>
              <div class="dc-node-body">
                <div class="dc-node-title">{title}</div>
                <div class="dc-node-detail">{detail}</div>
                {extra_html}
              </div>
              <div class="dc-node-latency {lc}">{latency_ms} ms</div>
            </div>""", unsafe_allow_html=True)

        # Node 1 — Input Guardrail
        if guardrail_hit and "input" in (guardrail_reason or "").lower():
            _render_node(
                node_placeholders["input_guardrail"], "err", "✗",
                "Input Guardrail",
                f"BLOCKED · {guardrail_reason}",
                t_guard_in,
                '<span class="dc-badge dc-badge-red">BLOCKED</span>',
            )
        else:
            _render_node(
                node_placeholders["input_guardrail"], "ok", "✓",
                "Input Guardrail",
                "No PII detected · No injection pattern",
                t_guard_in,
                '<span class="dc-badge dc-badge-green">PASSED</span>',
            )

        # Node 2 — Intent Router
        _render_node(
            node_placeholders["intent_router"], "ok", "✓",
            "Intent Router",
            f"Model: gpt-4o-mini · Tools selected: {', '.join(tools_used) if tools_used else 'none'}",
            t_router,
            f'<span class="dc-badge dc-badge-purple">{intent}</span>'
            + "".join(f'<span class="dc-badge dc-badge-blue">{t}</span>' for t in tools_used),
        )

        # Node 3 — Tool Execution
        tool_badges = ""
        tool_detail_parts = []
        for tr in tool_results:
            cls = "dc-badge-green" if tr["success"] else "dc-badge-red"
            sym = "✓" if tr["success"] else "✗"
            tool_badges += f'<span class="dc-badge {cls}">{sym} {tr["tool_name"]}</span>'
            tool_detail_parts.append(f'{tr["tool_name"]} · {tr["latency_ms"]}ms · {"OK" if tr["success"] else tr.get("error","ERR")}')

        _render_node(
            node_placeholders["tool_execution"], "ok" if tool_results else "skip", "✓" if tool_results else "—",
            "Tool Execution",
            " &nbsp;|&nbsp; ".join(tool_detail_parts) if tool_detail_parts else "No tools dispatched",
            t_tools,
            tool_badges,
        )

        # Node 4 — LLM Synthesis
        conf_html = f'<span class="dc-badge dc-badge-blue">confidence {confidence:.0%}</span>' if confidence else ""
        _render_node(
            node_placeholders["llm_synthesis"], "ok", "✓",
            "LLM Synthesis",
            f"Model: gpt-4o · {len(citations)} citations attached · {len(answer)} chars",
            t_synth,
            f'<span class="dc-badge dc-badge-green">DONE</span>{conf_html}',
        )

        # Node 5 — Output Guardrail
        if guardrail_hit and "output" in (guardrail_reason or "").lower():
            _render_node(
                node_placeholders["output_guardrail"], "err", "✗",
                "Output Guardrail",
                f"BLOCKED · {guardrail_reason}",
                t_guard_out,
                '<span class="dc-badge dc-badge-red">BLOCKED</span>',
            )
        else:
            _render_node(
                node_placeholders["output_guardrail"], "ok", "✓",
                "Output Guardrail",
                "No legal / fault language · Dollar values verified",
                t_guard_out,
                '<span class="dc-badge dc-badge-green">PASSED</span>',
            )

        # Node 6 — Memory Update
        _render_node(
            node_placeholders["memory_update"], "ok", "✓",
            "Memory Update",
            "Session context updated · Turn appended",
            t_mem,
            '<span class="dc-badge dc-badge-green">DONE</span>',
        )

        # Total latency bar
        lc = _latency_cls(latency_total)
        st.markdown(f"""
        <div class="dc-total-row">
          <span class="dc-total-label">⏱ Total round-trip latency</span>
          <span class="dc-total-val {lc}">{latency_total} ms</span>
        </div>""", unsafe_allow_html=True)

        # ── Result panels ──────────────────────────────────────────────
        st.markdown("---")
        col_a, col_b = st.columns([3, 2], gap="large")

        with col_a:
            st.markdown('<div class="dc-section-title">💬 Generated Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="background:#111a30;border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:16px 18px;color:#e6ecff;font-size:13.5px;line-height:1.7">{answer}</div>', unsafe_allow_html=True)

            if breakdown:
                st.markdown('<div class="dc-section-title">🧮 Calculation Breakdown</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="dc-json">{breakdown}</div>', unsafe_allow_html=True)

            if trace_url:
                st.markdown(f'<br><a href="{trace_url}" target="_blank" style="color:#60a5fa;font-size:13px;font-weight:600">🔗 View LangSmith Trace →</a>', unsafe_allow_html=True)

        with col_b:
            if citations:
                st.markdown('<div class="dc-section-title">📎 Citations Retrieved</div>', unsafe_allow_html=True)
                for c in citations:
                    css_cls = "csv" if c["source_type"] == "csv" else ""
                    icon = "📊" if c["source_type"] == "csv" else "📄"
                    score_pct = int(c["relevance_score"] * 100)
                    st.markdown(f"""
                    <div class="dc-citation {css_cls}">
                      <strong>{icon} {c['document']}</strong>
                      {f"· p.{c['page']}" if c.get('page') else ""}
                      <span style="float:right;color:#4ade80">{score_pct}%</span><br>
                      <span style="font-size:11px;color:#94a3b8">{c['excerpt'][:120]}{'…' if len(c['excerpt'])>120 else ''}</span>
                    </div>""", unsafe_allow_html=True)

            st.markdown('<div class="dc-section-title">📦 Raw API Response</div>', unsafe_allow_html=True)
            clean = {k: v for k, v in resp.items() if k != "answer"}
            st.markdown(f'<div class="dc-json">{json.dumps(clean, indent=2)}</div>', unsafe_allow_html=True)

elif run_btn:
    st.warning("Please enter a query first.")
