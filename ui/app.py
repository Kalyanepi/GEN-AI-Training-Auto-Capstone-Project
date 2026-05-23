"""RoadGuard AI Copilot — Streamlit entry point (v3 UI).

Professional design system with a neon-glow RoadGuard logo as the brand
signature. Backend wiring (FastAPI /api/v1/chat contract) is unchanged.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from ui.utils.api_client import get_api_base_url
from ui.utils.session_state import init_session_state
from ui.components.sidebar import render_sidebar
from ui.components.chat_panel import render_chat_panel, render_chat_input, handle_faq_query_param
from ui.components.right_panel import render_right_panel, handle_rp_query_param

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="RoadGuard AI Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

# ── Theme tokens ──────────────────────────────────────────────────────
_dark = st.session_state.get("ui_theme", "dark") == "dark"

if _dark:
    _THEME_VARS = """
      --bg:#0b1220; --sidebar-bg:#06142b; --sidebar-bg2:#0a1b37;
      --surface:#111a30; --surface2:#162243; --surface3:#1b2a52;
      --border:rgba(255,255,255,.07); --border2:rgba(255,255,255,.12);
      --text:#e6ecff; --text-2:#aab4cf; --text-3:#7c87a3;
      --sidebar-text:#f4f7ff; --sidebar-muted:#96a4be;
      --msg-user-bg:#1d2a4a; --msg-user-border:rgba(255,255,255,.08);
      --msg-ai:#111a30; --input-bg:#111a30;
      --shadow:0 18px 45px rgba(0,0,0,.45);
      --neon-strength:.9;
    """
else:
    _THEME_VARS = """
      --bg:#f8fafc; --sidebar-bg:#06142b; --sidebar-bg2:#0a1b37;
      --surface:#ffffff; --surface2:#f7f8fc; --surface3:#eef2fb;
      --border:rgba(15,23,42,.08); --border2:rgba(15,23,42,.14);
      --text:#10182f; --text-2:#596579; --text-3:#8a95aa;
      --sidebar-text:#f4f7ff; --sidebar-muted:#96a4be;
      --msg-user-bg:#172238; --msg-user-border:rgba(15,23,42,.08);
      --msg-ai:#ffffff; --input-bg:#ffffff;
      --shadow:0 18px 45px rgba(18,34,62,.10);
      --neon-strength:.55;
    """

# ── Neon RoadGuard mark (SVG glow filter + brand gradient) ────────────
_LOGO_SVG = """<svg class="rg-logo-svg" width="30" height="30" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="rgShield" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#a99cff"/><stop offset=".55" stop-color="#735df6"/><stop offset="1" stop-color="#4f3fd7"/>
    </linearGradient>
    <filter id="rgGlow" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="2.1" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <g filter="url(#rgGlow)">
    <path d="M16 2.4 4.4 7.3V16.2C4.4 22.9 9.6 28.7 16 30.5 22.4 28.7 27.6 22.9 27.6 16.2V7.3Z" fill="url(#rgShield)"/>
    <path d="M11.3 16.4l3.2 3.1L20.8 12" stroke="#fff" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  </g>
</svg>"""

# ── API health ────────────────────────────────────────────────────────
# WHY cached: Streamlit reruns the script top-to-bottom on every interaction.
# Without caching, each keystroke fires a fresh /health HTTP call (1.5s timeout)
# and adds latency to the entire UI.
@st.cache_data(ttl=10, show_spinner=False)
def _check_api_health(base_url: str) -> bool:
    try:
        import httpx as _hx
        _r = _hx.get(f"{base_url.rstrip('/')}/health", timeout=1.5)
        return _r.status_code == 200
    except Exception:
        return False

_api_ok = _check_api_health(get_api_base_url())

_api_html = (
    "<span class='rg-avatar-dot'></span>"
    if _api_ok else
    "<span class='rg-avatar-dot err-dot'></span>"
)

_theme_icon  = "☀" if _dark else "🌙"
_theme_label = "Light mode" if _dark else "Dark mode"
_next_theme  = "light" if _dark else "dark"

# ── Tabler Icons webfont + Inter ──────────────────────────────────────
st.markdown(
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/dist/tabler-icons.min.css">'
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ── CSS design system ─────────────────────────────────────────────────
st.markdown(f"""<style>
:root {{
  {_THEME_VARS}
  --accent:#6d5dfc; --accent-h:#816fff; --accent-dim:rgba(109,93,252,.12);
  --green:#10b981;  --green-dim:rgba(16,185,129,.14);
  --amber:#f59e0b;  --red:#ef4444; --blue:#60a5fa;
  --radius:16px; --radius-sm:12px; --radius-xs:8px;
  --trans:.18s cubic-bezier(.4,0,.2,1);
  --nav-h:68px; --side-w:336px;
}}

/* hide Streamlit chrome */
header[data-testid="stHeader"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stToolbar"],
#MainMenu, footer, .stDeployButton {{ display:none !important; }}
[data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] {{ display:none !important; }}

/* base */
.stApp, .stApp > .main, [data-testid="stAppViewContainer"] {{ background:var(--bg) !important; }}
.main .block-container {{ padding:0 !important; max-width:100% !important; }}
body, .stMarkdown, p, div, span, label, input, textarea, button {{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important;
  -webkit-font-smoothing:antialiased;
}}
::-webkit-scrollbar {{ width:7px; height:7px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:var(--border2); border-radius:6px; }}
::-webkit-scrollbar-thumb:hover {{ background:var(--text-3); }}

/* push content below fixed nav */
[data-testid="stSidebar"] {{ background:linear-gradient(180deg,var(--sidebar-bg),#020a18) !important; padding-top:var(--nav-h) !important; }}
[data-testid="stSidebar"] > div:first-child {{
  background:transparent !important;
  border-right:1px solid rgba(255,255,255,.08) !important; padding-top:92px !important;
  overflow:hidden !important;
}}
section.main > div.block-container {{ padding:calc(var(--nav-h) + 20px) 44px 72px 44px !important; }}

/* ───────────────────────── TOP NAV ───────────────────────── */
.rg-topnav {{
  position:fixed; top:0; left:0; right:0; height:var(--nav-h);
  background:rgba(255,255,255,.88);
  backdrop-filter:saturate(140%) blur(14px);
  border-bottom:1px solid var(--border);
  display:flex; align-items:center; z-index:9999;
}}
.rg-nav-brand {{
  width:var(--side-w); flex-shrink:0; height:100%;
  display:flex; align-items:center; gap:11px; padding:0 18px;
  background:linear-gradient(180deg,var(--sidebar-bg),var(--sidebar-bg2));
  border-right:1px solid rgba(255,255,255,.08);
}}
.rg-logo-svg {{ filter:drop-shadow(0 0 10px rgba(109,93,252,calc(var(--neon-strength) * .85))); }}
.rg-brand-text {{ display:flex; flex-direction:column; line-height:1.12; }}
.rg-brand-name {{
  font-weight:800; font-size:18px; letter-spacing:-.2px; color:var(--sidebar-text);
}}
.rg-brand-sub {{ display:none; }}
.rg-nav-center {{ flex:1; display:flex; align-items:center; justify-content:center; gap:9px; }}
.rg-nav-tag {{ display:none; }}
.rg-nav-tag i {{ color:var(--accent); margin-right:4px; opacity:.9; }}
.rg-nav-dot {{ display:none; }}
.rg-nav-right {{ display:flex; align-items:center; gap:16px; padding-right:28px; }}
.rg-nav-icon {{
  width:36px; height:36px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  color:#111827; text-decoration:none; border:1px solid transparent; transition:all var(--trans);
}}
.rg-nav-icon:hover {{ background:var(--surface2); border-color:var(--border); }}
.rg-deploy {{
  height:38px; padding:0 24px; display:flex; align-items:center; justify-content:center;
  background:#07142b; color:#fff; border-radius:8px; font-size:14px; font-weight:700;
  box-shadow:0 10px 24px rgba(7,20,43,.22);
}}
.rg-avatar {{
  width:38px; height:38px; border-radius:50%; background:#ede7ff; color:#6d5dfc; font-weight:800;
  display:flex; align-items:center; justify-content:center; position:relative; font-size:13px;
}}
.rg-avatar-dot {{
  width:8px; height:8px; background:#30c46c; border-radius:50%; border:2px solid #fff;
  position:absolute; right:1px; bottom:2px;
}}

/* API badge */
.rg-api-badge {{
  display:flex; align-items:center; gap:6px; padding:5px 12px; border-radius:20px;
  font-size:11.5px; font-weight:600; border:1px solid rgba(16,185,129,.3);
  background:rgba(16,185,129,.1); color:#10b981; cursor:default;
}}
.rg-api-badge.err {{ border-color:rgba(239,68,68,.3); background:rgba(239,68,68,.1); color:#ef4444; }}
.rg-api-dot {{ width:7px; height:7px; border-radius:50%; background:#10b981; animation:apiPulse 2.4s ease infinite; }}
.err-dot {{ background:#ef4444; animation:none !important; }}
@keyframes apiPulse {{ 0%,100%{{box-shadow:0 0 0 0 rgba(16,185,129,.5)}} 50%{{box-shadow:0 0 0 5px rgba(16,185,129,0)}} }}

/* settings dropdown (pure details/summary) */
.rg-settings-wrap {{ position:relative; list-style:none; }}
.rg-settings-wrap summary {{ list-style:none; }}
.rg-settings-wrap summary::-webkit-details-marker {{ display:none; }}
.rg-settings-btn {{
  width:36px; height:36px; border-radius:12px; border:1px solid transparent;
  background:transparent; display:flex; align-items:center; justify-content:center;
  cursor:pointer; color:#111827; font-size:18px; user-select:none; transition:all var(--trans);
}}
.rg-settings-btn:hover {{ background:var(--surface2); border-color:var(--border); }}
.rg-dropdown {{
  position:absolute; top:calc(100% + 9px); right:0; width:216px;
  background:var(--surface); border:1px solid var(--border2); border-radius:var(--radius-sm);
  box-shadow:var(--shadow); z-index:10000; overflow:hidden; display:none; animation:ddIn .14s ease;
}}
.rg-settings-wrap[open] .rg-dropdown {{ display:block; }}
@keyframes ddIn {{ from{{opacity:0;transform:translateY(-6px)}} to{{opacity:1;transform:translateY(0)}} }}
.rg-dd-hdr {{
  padding:10px 14px 7px; font-size:10px; font-weight:700; text-transform:uppercase;
  letter-spacing:.9px; color:var(--text-3); border-bottom:1px solid var(--border);
}}
.rg-dd-item {{
  display:flex; align-items:center; gap:9px; padding:10px 14px; cursor:pointer;
  font-size:12.5px; color:var(--text-2); transition:background var(--trans); text-decoration:none;
}}
.rg-dd-item:hover {{ background:var(--surface2); color:var(--text); }}

/* ───────────────────────── SIDEBAR ───────────────────────── */
.rg-s-label {{
  font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:1px;
  color:var(--sidebar-muted); margin-bottom:9px; padding:0 8px; display:block;
}}
.rg-side-link {{
  display:flex; align-items:center; gap:13px; padding:12px 16px; margin:4px 0; border-radius:9px;
  color:var(--sidebar-text); font-weight:600; font-size:14px; text-decoration:none;
}}
.rg-sidebar-brand {{
  position:fixed; top:0; left:0; width:336px; height:68px; z-index:1000000;
  display:flex; align-items:center; gap:12px; padding:0 22px;
  background:linear-gradient(180deg,var(--sidebar-bg),var(--sidebar-bg2));
  border-right:1px solid rgba(255,255,255,.08);
}}
.rg-sidebar-mark {{
  width:34px; height:34px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#a99cff,#5b49df); color:#fff;
  box-shadow:0 0 20px rgba(109,93,252,.45);
}}
.rg-sidebar-brand-text {{ font-size:18px; font-weight:800; color:var(--sidebar-text); letter-spacing:0; }}
.rg-side-link i {{ color:var(--sidebar-muted); font-size:17px; }}
.rg-side-link.active {{ background:rgba(255,255,255,.09); box-shadow:inset 0 0 0 1px rgba(255,255,255,.04); }}
.rg-side-link.active i {{ color:#8b7cff; }}
.rg-sidebar-spacer {{ height:24px; }}
.rg-tier {{ padding:12px 13px; border-radius:var(--radius-sm); border:1px solid rgba(255,255,255,.09); margin-top:8px; }}
.rg-tier.active {{ background:rgba(109,93,252,.16); border-color:rgba(109,93,252,.35); }}
.rg-tier-top {{ display:flex; align-items:center; gap:8px; margin-bottom:4px; }}
.rg-tier-name {{ font-size:13px; font-weight:700; color:var(--sidebar-text); flex:1; }}
.rg-tier-badge {{
  font-size:9px; font-weight:800; padding:2px 7px; border-radius:5px; text-transform:uppercase;
  letter-spacing:.5px; background:rgba(16,185,129,.18); color:#43d58c; border:1px solid rgba(16,185,129,.28);
}}
.rg-tier-ded {{ font-size:11px; color:var(--sidebar-muted); }}

.rg-stat-row {{ display:flex; justify-content:space-between; align-items:center; padding:4px 2px; }}
.rg-stat-lbl {{ font-size:12px; color:var(--text-3); }}
.rg-stat-val {{ font-size:12px; font-weight:600; color:var(--text); }}

/* recent conversations */
.rg-recent-item {{ display:flex; align-items:center; gap:11px; padding:11px 14px; border-radius:9px; transition:background var(--trans); color:var(--sidebar-text); }}
.rg-recent-item.active, .rg-recent-item:hover {{ background:rgba(255,255,255,.08); }}
.rg-recent-icon {{ font-size:14px; color:var(--sidebar-muted); flex-shrink:0; }}
.rg-recent-text {{ font-size:13px; color:var(--sidebar-text); line-height:1.35; font-weight:600; }}
.rg-recent-list {{
  max-height:calc(100vh - 430px); overflow-y:auto; padding-right:4px; margin-bottom:14px;
}}
.rg-recent-empty {{
  border:1px dashed rgba(255,255,255,.14); border-radius:10px; padding:14px;
  color:var(--sidebar-muted); font-size:13px;
}}
.rg-sidebar-bottom {{
  position:absolute; left:24px; right:24px; bottom:22px;
}}
.rg-health {{
  display:flex; align-items:center; gap:10px; height:42px; border-radius:12px; padding:0 14px;
  color:var(--sidebar-text); font-weight:800; font-size:13px; margin-bottom:12px;
  border:1px solid rgba(255,255,255,.12); background:rgba(8,24,52,.72);
}}
.rg-health-dot {{ width:10px; height:10px; border-radius:50%; }}
.rg-health span:last-child {{ margin-left:2px; }}
.rg-health.ok {{ box-shadow:0 0 24px rgba(16,185,129,.18); }}
.rg-health.ok .rg-health-dot {{ background:#10b981; box-shadow:0 0 12px #10b981; }}
.rg-health.bad {{ box-shadow:0 0 24px rgba(239,68,68,.24); animation:healthBad 1.1s ease-in-out infinite alternate; }}
.rg-health.bad .rg-health-dot {{ background:#ef4444; box-shadow:0 0 12px #ef4444; }}
@keyframes healthBad {{ from{{border-color:rgba(239,68,68,.24)}} to{{border-color:rgba(239,68,68,.75)}} }}

/* policy context card */
.rg-policy-card {{ background:rgba(8,24,52,.72); border:1px solid rgba(255,255,255,.13); border-radius:var(--radius); padding:17px 17px; }}
.rg-policy-card-header {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:11px; }}
.rg-policy-card-title {{ font-size:15px; font-weight:800; color:var(--sidebar-text); }}
.rg-policy-badge {{
  font-size:9px; font-weight:800; padding:2px 8px; border-radius:20px; background:var(--green-dim);
  color:#10b981; border:1px solid rgba(16,185,129,.28); text-transform:uppercase; letter-spacing:.4px;
}}
.rg-policy-row {{ display:flex; justify-content:space-between; align-items:center; padding:5px 0; border-bottom:1px solid var(--border); }}
.rg-policy-row:last-child {{ border-bottom:none; }}
.rg-policy-lbl {{ font-size:12px; color:var(--sidebar-muted); }}
.rg-policy-val {{ font-size:12.5px; font-weight:600; color:var(--sidebar-text); text-align:right; }}

/* tier pills */
[data-testid="stPillsBlock"] button {{ font-size:11.5px !important; border-radius:9px !important; padding:6px 11px !important; font-weight:600 !important; }}
[data-testid="stPillsBlock"] button[aria-pressed="true"] {{ background:var(--accent) !important; border-color:var(--accent) !important; color:#fff !important; }}

/* new-session button */
.rg-clear-wrap [data-testid="stButton"] > button {{
  width:100% !important; background:transparent !important; border:1px solid var(--border2) !important;
  border-radius:var(--radius-sm) !important; color:var(--text-2) !important; font-size:12.5px !important;
  font-weight:600 !important; padding:9px 12px !important; transition:all var(--trans) !important;
}}
.rg-clear-wrap [data-testid="stButton"] > button:hover {{
  background:rgba(239,68,68,.08) !important; border-color:rgba(239,68,68,.32) !important; color:#ef4444 !important;
}}

/* ───────────────────────── WELCOME ───────────────────────── */
.rg-welcome {{ display:flex; flex-direction:column; align-items:center; padding:18px 0 6px; gap:9px; }}
.rg-welcome-badge {{
  width:58px; height:58px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#8067ff,#5b49df); color:white; box-shadow:0 18px 36px rgba(109,93,252,.28);
  font-size:25px;
}}
.rg-welcome-title {{ font-size:30px; font-weight:800; letter-spacing:0; color:var(--text); text-align:center; }}
.rg-welcome-title .hl {{ color:var(--text); }}
.rg-welcome-sub {{ font-size:15px; color:var(--text-2); text-align:center; max-width:520px; line-height:1.65; }}
.rg-feat-row {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:center; margin-top:4px; }}
.rg-feat-tag {{
  font-size:11px; color:var(--text-2); background:var(--surface2); border:1px solid var(--border);
  border-radius:20px; padding:5px 13px; display:inline-flex; align-items:center; gap:6px;
}}
.rg-feat-tag i {{ color:var(--accent); }}
.rg-faq-label {{
  font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:1px;
  color:var(--text-3); text-align:center; margin:12px 0 10px;
}}

/* FAQ cards */
.rg-faq-card {{
  display:flex; align-items:flex-start; gap:12px; padding:14px 15px; background:var(--surface);
  border:1px solid var(--border); border-radius:var(--radius-sm); margin-bottom:9px;
  text-decoration:none !important; transition:all var(--trans); color:inherit !important;
}}
.rg-faq-card:hover {{ border-color:rgba(109,93,252,.35); background:var(--surface2); transform:translateY(-1px); box-shadow:var(--shadow); }}
.rg-faq-icon-wrap {{
  width:32px; height:32px; border-radius:9px; background:var(--accent-dim); display:inline-flex;
  align-items:center; justify-content:center; flex-shrink:0; font-size:15px; color:var(--accent);
}}
.rg-faq-body {{ display:inline-flex; flex-direction:column; gap:3px; }}
.rg-faq-cat {{ font-size:10px; font-weight:800; color:var(--accent); text-transform:uppercase; letter-spacing:.6px; }}
.rg-faq-q {{ font-size:12.5px; color:var(--text); line-height:1.5; font-weight:500; }}

/* ───────────────────────── MESSAGES (Claude-style) ───────────────────────── */
/* Each chat turn is its own flex row. User → right, assistant → left/full-width. */
.rg-turn {{
  display:flex; flex-direction:column; width:100%;
  animation:msgIn .22s cubic-bezier(.4,0,.2,1) both;
  margin:14px 0;
}}
.rg-turn-user {{ align-items:flex-end; }}
.rg-turn-ai   {{ align-items:stretch; }}
@keyframes msgIn {{
  from {{ opacity:0; transform:translateY(6px); }}
  to   {{ opacity:1; transform:translateY(0); }}
}}

/* User bubble: dark pill, right aligned, no avatar. */
.rg-msg-user {{
  background:var(--msg-user-bg); border:1px solid var(--msg-user-border);
  border-radius:18px 18px 4px 18px; padding:13px 18px;
  font-size:14.5px; line-height:1.55; color:#fff;
  max-width:75%; box-shadow:var(--shadow);
  white-space:pre-wrap; word-wrap:break-word;
}}
.rg-ts-user {{ text-align:right; margin-right:4px; }}

/* Assistant content: full-width, plain text (no bubble), high contrast. */
.rg-msg-ai {{
  color:var(--text) !important;
  font-size:14.5px; line-height:1.7;
  padding:2px 0; max-width:100%;
}}
.rg-msg-ai p, .rg-msg-ai li, .rg-msg-ai strong, .rg-msg-ai em {{ color:var(--text) !important; }}
.rg-msg-ai h1, .rg-msg-ai h2, .rg-msg-ai h3, .rg-msg-ai h4 {{
  color:var(--text) !important; margin-top:14px; margin-bottom:6px; font-weight:700;
}}
.rg-msg-ai code {{
  background:var(--surface2); color:var(--text); padding:1px 6px; border-radius:5px;
  font-size:13px; border:1px solid var(--border);
}}
.rg-msg-ai a {{ color:#60a5fa; }}

.rg-guardrail {{
  background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.22);
  border-radius:12px; padding:12px 15px; color:var(--amber); font-size:13px; max-width:84%;
}}
.rg-breakdown {{
  margin-top:10px; background:var(--surface2); border:1px solid var(--border);
  border-left:3px solid var(--accent); border-radius:var(--radius-sm);
  padding:13px 15px; font-size:12px; color:var(--text-2); line-height:2;
}}
.rg-disclaimer {{
  margin-top:9px; padding:9px 13px; background:var(--surface2); border:1px solid var(--border);
  border-radius:var(--radius-xs); font-size:11.5px; color:var(--text-3); line-height:1.55;
}}
.rg-ts {{ font-size:10.5px; color:var(--text-3); margin-top:5px; }}

/* Thinking placeholder — animated dots + cycling label. */
.rg-thinking {{
  display:inline-flex; align-items:center; gap:11px;
  padding:11px 16px; background:var(--surface);
  border:1px solid var(--border); border-radius:14px;
  color:var(--text-2); font-size:13.5px; font-style:italic;
  box-shadow:var(--shadow);
}}
.rg-thinking-dots {{ display:inline-flex; gap:4px; align-items:center; }}
.rg-thinking-dots span {{
  width:7px; height:7px; border-radius:50%; background:var(--accent);
  display:inline-block; animation:thinkPulse 1.2s ease-in-out infinite;
}}
.rg-thinking-dots span:nth-child(2) {{ animation-delay:.18s; }}
.rg-thinking-dots span:nth-child(3) {{ animation-delay:.36s; }}
@keyframes thinkPulse {{
  0%, 80%, 100% {{ opacity:.25; transform:translateY(0); }}
  40%           {{ opacity:1;   transform:translateY(-3px); }}
}}
.rg-thinking-text {{ color:var(--text-2); }}

/* citations */
.rg-citation-pdf {{ background:rgba(96,165,250,.07); border-left:3px solid var(--blue); border-radius:4px; padding:9px 11px; font-size:12px; color:var(--text-2); font-style:italic; }}
.rg-citation-csv {{ background:rgba(16,185,129,.07); border-left:3px solid var(--green); border-radius:4px; padding:9px 11px; font-size:12px; color:var(--text-2); font-style:italic; }}

/* metric pills */
.rg-metrics {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; align-items:center; }}
.rg-pill {{ font-size:10.5px; font-weight:600; padding:3px 9px; border-radius:20px; border:1px solid; display:inline-block; }}
.rg-pill-green  {{ color:#10b981; border-color:rgba(16,185,129,.3); background:rgba(16,185,129,.08); }}
.rg-pill-orange {{ color:#f59e0b; border-color:rgba(245,158,11,.3); background:rgba(245,158,11,.08); }}
.rg-pill-red    {{ color:#ef4444; border-color:rgba(239,68,68,.3);  background:rgba(239,68,68,.08); }}
.rg-pill-blue   {{ color:#60a5fa; border-color:rgba(96,165,250,.3); background:rgba(96,165,250,.08); }}
.rg-pill-gray   {{ color:var(--text-3); border-color:var(--border2); background:var(--surface2); }}

/* LangSmith trace button — blue pill with arrow */
.rg-langsmith-btn {{
  display:inline-flex; align-items:center; gap:5px; padding:3px 11px;
  font-size:10.5px; font-weight:700; letter-spacing:.2px;
  background:linear-gradient(135deg,#3b82f6,#2563eb);
  color:#fff !important; text-decoration:none !important;
  border-radius:20px; border:1px solid rgba(37,99,235,.6);
  box-shadow:0 2px 8px rgba(37,99,235,.28);
  transition:all var(--trans);
}}
.rg-langsmith-btn:hover {{
  background:linear-gradient(135deg,#60a5fa,#3b82f6);
  box-shadow:0 4px 14px rgba(37,99,235,.45);
  transform:translateY(-1px);
}}
.rg-langsmith-btn i {{ font-size:13px; }}

/* follow-up pills */
.rg-followup-wrap {{ display:flex; flex-wrap:wrap; gap:7px; margin-top:12px; }}
.rg-followup-wrap [data-testid="stButton"] > button {{
  background:transparent !important; border:1px solid var(--border2) !important; border-radius:20px !important;
  padding:6px 14px !important; font-size:12px !important; color:var(--text-2) !important; font-weight:500 !important;
  transition:all var(--trans) !important; white-space:normal !important; text-align:left !important;
}}
.rg-followup-wrap [data-testid="stButton"] > button:hover {{
  background:var(--surface2) !important; border-color:var(--accent) !important; color:var(--text) !important;
}}

/* context popup */
.rg-ctx {{
  background:var(--surface); border:1px solid var(--border2); border-radius:var(--radius);
  padding:15px 17px; margin-bottom:10px; box-shadow:var(--shadow); animation:ctxUp .18s ease;
}}
@keyframes ctxUp {{ from{{opacity:0;transform:translateY(6px)}} to{{opacity:1;transform:translateY(0)}} }}
.rg-ctx-label {{
  font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:.8px;
  color:var(--accent); margin-bottom:11px; display:flex; align-items:center; gap:7px;
}}
.rg-ctx-label::before {{ content:''; width:6px; height:6px; border-radius:50%; background:var(--accent); box-shadow:0 0 7px var(--accent); }}
.rg-ctx [data-testid="stNumberInput"] input,
.rg-ctx [data-testid="stSelectbox"] div[data-baseweb="select"] {{
  background:var(--surface2) !important; border-color:var(--border2) !important;
  color:var(--text) !important; border-radius:var(--radius-xs) !important;
}}
.rg-ctx label {{ color:var(--text-3) !important; font-size:11px !important; }}

/* ───────────────────────── RIGHT PANEL ───────────────────────── */
.rg-rp-header {{ font-size:13px; font-weight:700; color:var(--text); margin-bottom:13px; }}
.rg-rp-card {{
  display:flex; align-items:center; gap:13px; padding:17px 17px; background:var(--surface);
  border:1px solid var(--border); border-radius:10px; margin-bottom:12px;
  text-decoration:none !important; transition:all var(--trans); color:inherit !important;
}}
.rg-rp-card:hover {{ border-color:rgba(109,93,252,.34); background:var(--surface2); transform:translateX(2px); }}
.rg-rp-icon {{
  width:36px; height:36px; border-radius:9px; background:var(--accent-dim); display:inline-flex;
  align-items:center; justify-content:center; flex-shrink:0; font-size:15px; color:var(--accent);
}}
.rg-rp-body {{ display:inline-flex; flex-direction:column; flex:1; min-width:0; }}
.rg-rp-title {{ font-size:12.5px; font-weight:700; color:var(--text); }}
.rg-rp-sub {{ font-size:11px; color:var(--text-3); margin-top:1px; }}
.rg-rp-chev {{ font-size:14px; color:var(--text-3); flex-shrink:0; }}
[data-testid="column"]:last-of-type {{ border-left:1px solid var(--border); padding-left:18px !important; padding-top:6px !important; }}

/* ───────────────────────── CHAT INPUT ───────────────────────── */
[data-testid="stChatInput"] textarea {{
  background:var(--input-bg) !important; border:1px solid var(--border2) !important;
  border-radius:14px !important; color:var(--text) !important; font-size:13.5px !important;
}}
[data-testid="stChatInput"] textarea:focus {{
  border-color:rgba(109,93,252,.45) !important; box-shadow:0 0 0 3px rgba(109,93,252,.1) !important;
}}
[data-testid="stChatInputSubmitButton"], [data-testid="stChatInput"] button {{
  background:var(--accent) !important; border-radius:10px !important;
}}
[data-testid="stChatMessageAvatarUser"] {{ background:var(--surface3) !important; border:1px solid var(--border2) !important; border-radius:9px !important; }}
[data-testid="stChatMessageAvatarAssistant"] {{
  background:linear-gradient(135deg,#8067ff,#5b49df) !important; border-radius:9px !important;
  box-shadow:0 2px 12px rgba(109,93,252,calc(var(--neon-strength) * .6)) !important;
}}
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {{
  background:rgba(248,250,252,.96) !important; border-top:0 !important; padding:8px 18px 16px !important;
}}
[data-testid="stMain"], section[data-testid="stMain"] > div {{ background:var(--bg) !important; }}
.rg-input-note {{
  position:fixed; left:var(--side-w); right:0; bottom:2px; text-align:center; color:var(--text-2);
  font-size:12px; z-index:1000; pointer-events:none;
}}
</style>""", unsafe_allow_html=True)

# ── Top-nav HTML ──────────────────────────────────────────────────────
st.markdown(f"""
<div class="rg-topnav">
  <div class="rg-nav-brand">
    {_LOGO_SVG}
      <div class="rg-brand-text">
      <div class="rg-brand-name">RoadGuard AI</div>
      <div class="rg-brand-sub">Auto Insurance Copilot</div>
    </div>
  </div>
  <div class="rg-nav-center">
    <span class="rg-nav-tag"><i class="ti ti-circle-check"></i>Accurate</span><span class="rg-nav-dot"></span>
    <span class="rg-nav-tag"><i class="ti ti-anchor"></i>Grounded</span><span class="rg-nav-dot"></span>
    <span class="rg-nav-tag"><i class="ti ti-quote"></i>Cited</span><span class="rg-nav-dot"></span>
    <span class="rg-nav-tag"><i class="ti ti-shield-lock"></i>Never Fabricated</span>
  </div>
  <div class="rg-nav-right">
    <details class="rg-settings-wrap">
      <summary class="rg-settings-btn"><i class="ti ti-settings"></i></summary>
      <div class="rg-dropdown">
        <div class="rg-dd-hdr">Preferences</div>
        <a class="rg-dd-item" href="?ui_theme={_next_theme}"><span>{_theme_icon}</span> {_theme_label}</a>
        <div class="rg-dd-hdr">About</div>
        <div class="rg-dd-item"><span>🛡️</span> RoadGuard v3.0</div>
      </div>
    </details>
    <div class="rg-avatar">AM{_api_html}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Query params ──────────────────────────────────────────────────────
_qp = st.query_params.to_dict()
if "ui_theme" in _qp:
    st.session_state.ui_theme = _qp["ui_theme"]
    st.query_params.clear()
    st.rerun()
if "tier" in _qp and _qp["tier"] in ("standard", "premium", "elite"):
    st.session_state.policy_tier = _qp["tier"]
    st.query_params.clear()
    st.rerun()
if "faq_idx" in _qp:
    handle_faq_query_param()
if "rp" in _qp:
    handle_rp_query_param()

# ── Render panels ─────────────────────────────────────────────────────
col_chat, col_right = st.columns([3, 1], gap="medium")
with col_chat:
    render_chat_panel()
with col_right:
    render_right_panel()
# Chat input MUST be outside columns — otherwise stBottom pushes right panel below.
render_chat_input()
render_sidebar()
