import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import re
import base64
import io
import hashlib
from pathlib import Path



from sklearn.utils.validation import check_is_fitted
from report_ai.ocr.reader import extract_text_from_image

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import fitz
except ImportError:
    fitz = None

from PIL import Image
from report_ai.chatbot import ask_nirmaya_ai
from patient_history import (
    create_database,
    save_patient_record,
    get_patient_history,
    get_patient_names,
    get_or_create_patient_id,
    get_patient_history_by_id,
    register_patient,
    register_doctor,
    login_user,
    get_patient_by_id,
    get_patient_by_name,
    save_patient_report,
    get_patient_reports,
    save_patient_message,
    get_patient_messages,
    get_unanswered_patient_messages,
    reply_to_patient_message
)
create_database()

st.set_page_config(
    page_title="NIRAMAYA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
/* ========================================================
   NIRAMAYA - LIGHT HEALTHCARE UI
   ======================================================== */
.stApp {
    background:
        radial-gradient(circle at 8% 18%, rgba(45, 212, 191, 0.10), transparent 28%),
        radial-gradient(circle at 92% 12%, rgba(56, 189, 248, 0.12), transparent 30%),
        linear-gradient(180deg, #f8fdff 0%, #eef9fc 48%, #ffffff 100%);
    color: #12345b;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
h1, h2, h3, h4 { color: #123b68 !important; }
p, label, .stMarkdown { color: #315579 !important; }

/* Top brand / navigation feel */
.niramaya-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    padding: 14px 24px;
    margin: 0 0 18px 0;
    background: rgba(255,255,255,0.90);
    border: 1px solid rgba(14, 165, 183, 0.16);
    border-radius: 18px;
    box-shadow: 0 8px 28px rgba(14, 116, 144, 0.10);
    backdrop-filter: blur(12px);
}
.niramaya-brand {
    display:flex; align-items:center; gap:12px;
    color:#073b68 !important;
    font-size:28px; font-weight:850; letter-spacing:-0.5px;
}
.niramaya-brand-icon {
    width:46px; height:46px; border-radius:14px;
    display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg,#0ea5a8,#14b8a6);
    color:white; font-size:26px;
    box-shadow:0 7px 18px rgba(20,184,166,.25);
}
.niramaya-nav {
    color:#174a73; font-size:14px; font-weight:650;
    display:flex; gap:22px; flex-wrap:wrap; justify-content:center;
}
.niramaya-nav span { padding:8px 2px; }
.niramaya-nav .active { color:#08a6a5; border-bottom:3px solid #14b8a6; }

/* Hero */
.niramaya-hero {
    position:relative; overflow:hidden;
    padding:42px 48px; margin-bottom:24px;
    border-radius:28px;
    background:linear-gradient(115deg,#ffffff 0%,#eefcff 48%,#d9f6fb 100%);
    border:1px solid rgba(14,165,183,.16);
    box-shadow:0 14px 38px rgba(14,116,144,.12);
}
.niramaya-hero:after {
    content:""; position:absolute; width:360px; height:360px;
    right:-100px; top:-120px; border-radius:50%;
    background:radial-gradient(circle,rgba(45,212,191,.20),transparent 68%);
}
.niramaya-kicker {
    display:inline-block; padding:8px 16px; border-radius:999px;
    background:#e7fbf8; color:#078b8c; font-weight:700; font-size:14px;
    border:1px solid #b8eee7;
}
.niramaya-hero h1 {
    font-size:48px; line-height:1.06; margin:18px 0 14px;
    letter-spacing:-1.8px;
}
.niramaya-hero .gradient-text {
    background:linear-gradient(90deg,#0f766e,#0891b2,#2563eb);
    -webkit-background-clip:text; background-clip:text; color:transparent !important;
}
.niramaya-hero p { max-width:650px; font-size:17px; line-height:1.7; }
.niramaya-hero-btn {
    display:inline-block; margin-top:18px; padding:14px 24px;
    border-radius:999px; color:white !important; font-weight:750;
    background:linear-gradient(90deg,#0891b2,#14b8a6);
    box-shadow:0 9px 24px rgba(8,145,178,.24);
}

/* Inputs / controls */
.stTextInput input, .stNumberInput input,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background:#ffffff !important; color:#123b68 !important;
    border-radius:12px !important;
    border:1px solid #cfe5ed !important;
}
.stSelectbox div[data-baseweb="select"] * { color:#123b68 !important; }
.stButton > button {
    width:100%; border:none; border-radius:13px; padding:11px 20px;
    font-size:15px; font-weight:750; color:white !important;
    background:linear-gradient(90deg,#0891b2,#14b8a6);
    box-shadow:0 7px 20px rgba(8,145,178,.18);
}
.stButton > button:hover { transform:translateY(-2px); box-shadow:0 11px 26px rgba(8,145,178,.28); }
[data-testid="stFileUploader"] {
    background:rgba(255,255,255,.88); border:1px dashed #7dd3df;
    border-radius:18px; padding:15px;
}
.stRadio > div { background:rgba(255,255,255,.78); border-radius:14px; padding:10px 15px; }
[data-testid="stMetric"] {
    background:rgba(255,255,255,.92); border:1px solid #d4eaf0;
    border-radius:18px; padding:18px; box-shadow:0 6px 20px rgba(14,116,144,.07);
}
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#ffffff 0%,#edfaff 100%);
    border-right:1px solid #d7edf2;
}
section[data-testid="stSidebar"] * { color:#173f64 !important; }
hr { border-color:#d9edf2; }
.stCaption { color:#68869d !important; }

/* Section headings */
.niramaya-section-title {
    text-align:center; font-size:30px; font-weight:850; color:#123f70;
    margin:14px 0 4px;
}
.nir-section-title-highlight {
    display: inline-block;
    padding: 12px 22px;
    margin: 18px 0 14px;
    border-radius: 14px;
    background: linear-gradient(135deg, #e6fffb, #dbeafe);
    border: 1px solid rgba(13, 148, 136, 0.28);
    box-shadow: 0 8px 22px rgba(15, 118, 110, 0.12);
    color: #0f766e;
    font-size: 24px;
    font-weight: 800;
}

.niramaya-section-subtitle {
    text-align:center; color:#5d7890; margin-bottom:20px;
}

/* Chatbot */
.nirmaya-chat-box {
    background:linear-gradient(145deg,#ffffff,#effcff);
    border:1px solid #b8e7ed; border-radius:22px; padding:24px;
    box-shadow:0 12px 32px rgba(14,116,144,.13);
}
div[data-testid="stTextArea"] textarea {
    border-radius:14px !important; border:1px solid #9ddde5 !important;
    background:#ffffff !important; color:#123b68 !important;
}

/* Alerts */
.stAlert { border-radius:14px !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# NIRAMAYA HOME / TOP NAVIGATION
# ============================================================

# Hide Streamlit's default chrome so the page matches the NIRAMAYA design.
st.markdown("""
<style>
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
}
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding-top: 0.4rem !important; max-width: 1536px !important; }
</style>
""", unsafe_allow_html=True)

# Top navigation buttons. They keep the existing Prediction/About functionality
# while presenting it like the reference website.
nav_cols = st.columns([2.8, 1, 1, 1.15, 1, 1.25, 1, 1.35, 0.55])
with nav_cols[0]:
    st.markdown("""
    <div class="nir-brand">
        <div class="nir-shield">✚</div>
        <div><div class="nir-name">NIRAMAYA</div><div class="nir-tag">AI-Powered Early Disease Detection</div></div>
    </div>
    """, unsafe_allow_html=True)

nav_items = [(1, "⌂", "Home", "Home"), (2, "ⓘ", "About Us", "About System"),
             (3, "⚙", "How It Works", "How It Works"), (4, "♡", "Risk Checks", "Prediction"),
             (5, "▥", "Health Insights", "Health Insights"), (6, "✉", "Contact Us", "Contact Us")]
for idx, icon, label, target in nav_items:
    with nav_cols[idx]:
        if st.button(f"{icon}  {label}", key=f"topnav_{idx}", use_container_width=True):
            st.session_state["page"] = target
            st.rerun()
with nav_cols[7]:
    if st.button("♙  Login / Sign Up", key="topnav_login", use_container_width=True):
        st.session_state["page"] = "Prediction"
        st.rerun()
with nav_cols[8]:
    st.markdown('<div class="theme-dot">☾</div>', unsafe_allow_html=True)

st.markdown("""
<style>
.nir-brand{display:flex;align-items:center;gap:11px;padding:8px 0 10px 4px;white-space:nowrap}
.nir-shield{width:48px;height:54px;display:flex;align-items:center;justify-content:center;background:linear-gradient(150deg,#08b5ad,#1267a8);color:#fff;font-size:28px;clip-path:polygon(50% 0,94% 14%,88% 70%,50% 100%,12% 70%,6% 14%);box-shadow:0 7px 18px rgba(8,181,173,.22)}
.nir-name{font-size:28px;font-weight:900;letter-spacing:.5px;background:linear-gradient(90deg,#7fffea,#5bd8ff,#a78bfa);-webkit-background-clip:text;background-clip:text;color:transparent!important;line-height:1}
.nir-tag{font-size:11px;color:#8fb5c8;margin-top:5px}
.theme-dot{width:42px;height:42px;border:1px solid rgba(125,211,252,.20);border-radius:50%;display:flex;align-items:center;justify-content:center;background:rgba(10,28,43,.72);color:#d9f7ff;font-size:19px;margin-top:7px;box-shadow:0 8px 24px rgba(0,0,0,.18)}
div[data-testid="stHorizontalBlock"] button{border:1px solid transparent!important;background:rgba(8,26,42,.50)!important;color:#b8d6e5!important;font-weight:700!important;border-radius:12px!important;min-height:48px!important;padding:6px 7px!important}
div[data-testid="stHorizontalBlock"] button:hover{color:#7fffea!important;background:rgba(20,184,166,.10)!important;border-color:rgba(20,184,166,.16)!important}
</style>
""", unsafe_allow_html=True)


# ============================================================
# NIRMAYA GLOBAL DARK UI OVERRIDES
# ============================================================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: #06131f !important;
}
.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(20,184,166,.16), transparent 25%),
        radial-gradient(circle at 92% 10%, rgba(59,130,246,.16), transparent 28%),
        radial-gradient(circle at 50% 95%, rgba(124,58,237,.12), transparent 32%),
        linear-gradient(135deg, #04111c 0%, #071c2b 50%, #090f24 100%) !important;
    color: #e6f7ff !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#061726,#07111e) !important;
    border-right: 1px solid rgba(94,234,212,.10);
}
label, .stMarkdown, .stCaption, p, span { color: #c7ddea !important; }
.stTextInput input, .stTextArea textarea, .stNumberInput input,
[data-baseweb="select"] > div, [data-baseweb="input"] > div {
    background: rgba(13,31,47,.96) !important;
    color: #effaff !important;
    border: 1px solid rgba(125,211,252,.22) !important;
    border-radius: 13px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
    border-color: #2dd4bf !important;
    box-shadow: 0 0 0 2px rgba(45,212,191,.12) !important;
}
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
    background: #0b2030 !important;
    color: #e6f7ff !important;
}
[data-baseweb="option"] { color: #dff8ff !important; }
[data-baseweb="option"]:hover { background: rgba(45,212,191,.16) !important; }
[data-testid="stFileUploader"] {
    background: rgba(9,27,42,.85) !important;
    border: 1px dashed rgba(94,234,212,.35) !important;
    border-radius: 18px !important;
}
[data-testid="stFileUploaderDropzone"] { background: rgba(10,30,46,.80) !important; }
.stButton > button {
    background: linear-gradient(100deg,#10bfa8,#148cff 58%,#7655f5) !important;
    color: white !important;
    border: 0 !important;
    border-radius: 13px !important;
    font-weight: 800 !important;
    box-shadow: 0 8px 25px rgba(20,140,255,.16) !important;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 12px 30px rgba(45,212,191,.20) !important; }
[data-testid="stMetric"] {
    background: linear-gradient(145deg,rgba(12,39,55,.88),rgba(8,25,40,.92)) !important;
    border: 1px solid rgba(125,211,252,.14) !important;
    border-radius: 18px !important;
}
.doctor-reply-card {
    margin-top: 12px; padding: 16px 18px; border-radius: 16px;
    background: linear-gradient(135deg,rgba(20,184,166,.13),rgba(59,130,246,.12));
    border: 1px solid rgba(45,212,191,.28); color:#e8ffff;
}
.doctor-reply-card span { color:#dffaff !important; }
.doctor-reply-card small { color:#8fb5c8 !important; }
.patient-message-card {
    padding: 16px 18px; border-radius: 16px;
    background: linear-gradient(135deg,rgba(30,41,59,.90),rgba(15,35,52,.90));
    border:1px solid rgba(96,165,250,.20); margin-bottom:10px;
}
.patient-message-card b { color:#7fffea !important; }
.patient-message-card small { display:block; color:#86aabd !important; margin-top:3px; }
.patient-message-card p { color:#e2f1f7 !important; margin-top:10px; }
[data-testid="stTabs"] button { color:#b9d8e8 !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#5ff1d9 !important; }
hr { border-color: rgba(125,211,252,.12) !important; }
</style>
""", unsafe_allow_html=True)

page = st.session_state.get("page", "Home")

# ============================================================
# FLOATING NIRMAYA AI CHATBOT
# ============================================================
st.markdown("""
<style>
/* Floating AI launcher */
[data-testid="stPopover"] {
    position: fixed !important;
    right: 28px !important;
    bottom: 28px !important;
    z-index: 999999 !important;
}
[data-testid="stPopover"] > button {
    border-radius: 999px !important;
    min-height: 58px !important;
    min-width: 58px !important;
    width: 58px !important;
    padding: 0 !important;
    font-size: 25px !important;
    background: linear-gradient(135deg,#0ea5e9,#2563eb,#7c3aed) !important;
    color: white !important;
    border: 3px solid rgba(255,255,255,.9) !important;
    box-shadow: 0 10px 30px rgba(37,99,235,.30) !important;
}
[data-testid="stPopover"] > button:hover {
    transform: translateY(-3px) scale(1.04);
    box-shadow: 0 15px 38px rgba(37,99,235,.38) !important;
}
.nirmaya-floating-title {
    font-size: 22px;
    font-weight: 900;
    color: #075985 !important;
    margin-bottom: 2px;
}
.nirmaya-floating-subtitle {
    color: #52708d !important;
    font-size: 12px;
    margin-bottom: 12px;
}
.nirmaya-floating-answer {
    background: linear-gradient(135deg,#eff8ff,#eef2ff);
    border: 1px solid #bfdbfe;
    border-left: 4px solid #2563eb;
    border-radius: 14px;
    padding: 12px 14px;
    color: #163b61 !important;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

if "nirmaya_chat_answer" not in st.session_state:
    st.session_state["nirmaya_chat_answer"] = ""

with st.popover("🤖", use_container_width=False):
    st.markdown('<div class="nirmaya-floating-title">🤖 NIRMAYA AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nirmaya-floating-subtitle">Your intelligent healthcare assistant</div>',
        unsafe_allow_html=True
    )

    chatbot_question = st.text_area(
        "Ask NIRMAYA",
        placeholder="Ask about health parameters, reports, or your screening result...",
        key="nirmaya_chat_question",
        height=100,
        label_visibility="collapsed"
    )

    if st.button("🚀 Ask NIRMAYA AI", type="primary", key="nirmaya_chat_button"):
        if chatbot_question.strip():
            with st.spinner("🤖 NIRMAYA AI is thinking..."):
                try:
                    report_context = st.session_state.get("ocr_data", None)
                    answer = ask_nirmaya_ai(chatbot_question.strip(), report_context)
                    st.session_state["nirmaya_chat_answer"] = answer
                except Exception as e:
                    st.session_state["nirmaya_chat_answer"] = f"Unable to get an AI response: {e}"
        else:
            st.warning("Please enter a question first.")

    if st.session_state.get("nirmaya_chat_answer"):
        st.markdown(
            '<div class="nirmaya-floating-answer"><b>🤖 NIRMAYA AI</b></div>',
            unsafe_allow_html=True
        )
        st.markdown(st.session_state["nirmaya_chat_answer"])
        st.caption("⚠️ Informational screening support only. Not a medical diagnosis.")

# PREMIUM NIRMAYA HOME DESIGN
# ============================================================

st.markdown("""
<style>
/* ---------- Premium global surface ---------- */
.stApp {
    background:
        radial-gradient(circle at 5% 5%, rgba(20,184,166,.16), transparent 24%),
        radial-gradient(circle at 95% 8%, rgba(59,130,246,.15), transparent 25%),
        radial-gradient(circle at 50% 100%, rgba(124,58,237,.10), transparent 30%),
        linear-gradient(135deg, #06141f 0%, #071c2b 45%, #081326 100%) !important;
    color: #e8f7ff !important;
}

.block-container {
    max-width: 1500px !important;
    padding: .45rem 2rem 3rem !important;
}

/* ---------- Top navigation ---------- */
.nir-brand {
    display:flex;
    align-items:center;
    gap:12px;
    padding:7px 0 8px 4px;
    white-space:nowrap;
}

.nir-shield {
    width:48px;
    height:54px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:linear-gradient(145deg,#16e0c1,#1495ff 58%,#7047ff);
    color:#fff;
    font-size:27px;
    clip-path:polygon(50% 0,94% 14%,88% 70%,50% 100%,12% 70%,6% 14%);
    box-shadow:0 0 28px rgba(20,224,193,.32);
}

.nir-name {
    font-size:27px;
    font-weight:900;
    letter-spacing:.6px;
    line-height:1;
    background:linear-gradient(90deg,#7fffea,#5bd8ff,#a78bfa);
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent !important;
}

.nir-tag {
    font-size:10px;
    color:#8fb5c8;
    margin-top:5px;
    letter-spacing:.25px;
}

.theme-dot {
    width:42px;
    height:42px;
    border:1px solid rgba(125,211,252,.20);
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    background:rgba(10,28,43,.72);
    color:#d9f7ff;
    font-size:19px;
    margin-top:7px;
    box-shadow:0 8px 24px rgba(0,0,0,.18);
}

div[data-testid="stHorizontalBlock"] button {
    border:1px solid transparent !important;
    background:transparent !important;
    color:#b8d6e5 !important;
    font-weight:650 !important;
    border-radius:12px !important;
    min-height:46px !important;
    padding:6px 7px !important;
}

div[data-testid="stHorizontalBlock"] button:hover {
    color:#7fffea !important;
    background:rgba(20,184,166,.10) !important;
    border-color:rgba(20,184,166,.16) !important;
}

/* ---------- Hero ---------- */
.nir-home-hero {
    position:relative;
    overflow:hidden;
    min-height:520px;
    display:flex;
    align-items:stretch;
    margin:8px 0 20px;
    border:1px solid rgba(125,211,252,.16);
    border-radius:32px;
    background:
        linear-gradient(110deg, rgba(7,30,45,.97) 0%, rgba(7,28,44,.93) 47%, rgba(10,34,56,.82) 100%);
    box-shadow:
        0 25px 70px rgba(0,0,0,.35),
        inset 0 1px 0 rgba(255,255,255,.04);
}

.nir-home-hero:before {
    content:"";
    position:absolute;
    width:480px;
    height:480px;
    left:-190px;
    top:-230px;
    border-radius:50%;
    background:radial-gradient(circle, rgba(20,224,193,.22), transparent 68%);
}

.nir-home-hero:after {
    content:"";
    position:absolute;
    width:520px;
    height:520px;
    right:-220px;
    bottom:-290px;
    border-radius:50%;
    background:radial-gradient(circle, rgba(99,102,241,.22), transparent 68%);
}

.nir-home-left {
    width:55%;
    padding:68px 20px 58px 72px;
    position:relative;
    z-index:3;
}

.nir-ai-badge {
    display:inline-flex;
    align-items:center;
    gap:9px;
    padding:9px 15px;
    border-radius:999px;
    background:linear-gradient(90deg,rgba(20,224,193,.12),rgba(59,130,246,.13));
    border:1px solid rgba(94,234,212,.28);
    color:#8fffea;
    font-size:13px;
    font-weight:800;
    box-shadow:0 0 24px rgba(20,184,166,.08);
}

.nir-ai-dot {
    width:9px;
    height:9px;
    border-radius:50%;
    background:#38f2d2;
    box-shadow:0 0 14px #38f2d2;
}

.nir-home-title {
    font-size:58px !important;
    line-height:1.02 !important;
    letter-spacing:-2.8px !important;
    margin:20px 0 16px !important;
    color:#f3fbff !important;
    font-weight:900 !important;
}

.nir-home-title .mint {
    background:linear-gradient(90deg,#62f5da,#46d7ff);
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent !important;
}

.nir-home-title .violet {
    background:linear-gradient(90deg,#65dfff,#a78bfa);
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent !important;
}

.nir-home-description {
    max-width:650px;
    color:#a8c6d5;
    font-size:17px;
    line-height:1.72;
    margin-bottom:25px;
}

.nir-hero-pills {
    display:flex;
    flex-wrap:wrap;
    gap:9px;
    margin-bottom:27px;
}

.nir-hero-pill {
    padding:8px 12px;
    border-radius:999px;
    color:#c9e8f5;
    background:rgba(255,255,255,.045);
    border:1px solid rgba(148,163,184,.16);
    font-size:12px;
    font-weight:650;
}

.nir-home-visual {
    width:45%;
    position:relative;
    z-index:2;
    display:flex;
    align-items:center;
    justify-content:center;
    padding:30px 30px 30px 0;
}

.nir-home-visual img {
    width:100%;
    height:450px;
    object-fit:cover;
    object-position:center;
    border-radius:27px;
    opacity:.94;
    filter:saturate(1.08) contrast(1.04);
    box-shadow:0 25px 60px rgba(0,0,0,.28);
}

.nir-orbit-card {
    position:absolute;
    display:flex;
    align-items:center;
    gap:11px;
    padding:12px 15px;
    border-radius:18px;
    background:rgba(6,20,31,.78);
    border:1px solid rgba(125,211,252,.20);
    backdrop-filter:blur(14px);
    color:#dffaff;
    box-shadow:0 14px 35px rgba(0,0,0,.24);
}

.nir-orbit-card b { display:block; font-size:12px; }
.nir-orbit-card small { display:block; margin-top:3px; color:#8fb5c8; font-size:10px; }

.nir-orbit-icon {
    width:38px;
    height:38px;
    border-radius:12px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:linear-gradient(135deg,rgba(20,224,193,.18),rgba(59,130,246,.20));
    font-size:18px;
}

.nir-card-one { right:18px; top:65px; }
.nir-card-two { left:5px; bottom:75px; }
.nir-card-three { right:40px; bottom:25px; }

/* ---------- Streamlit CTA ---------- */
.nir-start-wrap {
    margin-top:0;
    margin-bottom:4px;
}

.nir-start-wrap .stButton > button {
    width:auto !important;
    min-width:225px !important;
    border:0 !important;
    border-radius:15px !important;
    padding:14px 25px !important;
    background:linear-gradient(100deg,#11cbb1,#149cff 58%,#7655f5) !important;
    color:#fff !important;
    font-size:15px !important;
    font-weight:850 !important;
    box-shadow:0 12px 32px rgba(20,184,166,.22) !important;
}

.nir-start-wrap .stButton > button:hover {
    transform:translateY(-2px);
    box-shadow:0 17px 38px rgba(59,130,246,.28) !important;
}

/* ---------- Trust / stats ---------- */
.nir-stats {
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:13px;
    margin:0 0 25px;
}

.nir-stat {
    padding:19px 20px;
    border-radius:20px;
    background:linear-gradient(145deg,rgba(12,39,55,.88),rgba(8,26,42,.82));
    border:1px solid rgba(125,211,252,.13);
    box-shadow:0 14px 32px rgba(0,0,0,.18);
}

.nir-stat-icon {
    font-size:21px;
    margin-bottom:8px;
}

.nir-stat b {
    display:block;
    color:#e8fbff;
    font-size:14px;
}

.nir-stat span {
    display:block;
    color:#82a9bb;
    font-size:11px;
    line-height:1.45;
    margin-top:5px;
}

/* ---------- Main feature cards ---------- */
.nir-feature-heading {
    text-align:center;
    margin:25px 0 5px;
    font-size:30px;
    font-weight:900;
    color:#edfaff;
}

.nir-feature-subtitle {
    text-align:center;
    color:#87a9bb;
    font-size:14px;
    margin-bottom:18px;
}

.nir-feature-grid {
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:15px;
    margin-bottom:25px;
}

.nir-feature-card {
    min-height:175px;
    padding:24px;
    border-radius:22px;
    background:linear-gradient(145deg,rgba(13,43,59,.90),rgba(8,27,44,.88));
    border:1px solid rgba(125,211,252,.13);
    box-shadow:0 15px 38px rgba(0,0,0,.18);
    transition:.2s ease;
}

.nir-feature-card:hover {
    transform:translateY(-4px);
    border-color:rgba(94,234,212,.32);
}

.nir-feature-icon {
    width:50px;
    height:50px;
    border-radius:15px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:23px;
    margin-bottom:14px;
    background:linear-gradient(135deg,rgba(20,224,193,.15),rgba(99,102,241,.17));
    border:1px solid rgba(125,211,252,.12);
}

.nir-feature-card h3 {
    margin:0 0 7px;
    color:#eaffff !important;
    font-size:17px;
}

.nir-feature-card p {
    margin:0;
    color:#91b1c0 !important;
    line-height:1.6;
    font-size:12px;
}

/* ---------- Disease strip ---------- */
.nir-disease-section {
    padding:25px 0 6px;
}

.nir-disease-grid {
    display:grid;
    grid-template-columns:repeat(6,1fr);
    gap:11px;
}

.nir-disease {
    text-align:center;
    padding:17px 9px;
    border-radius:18px;
    background:rgba(11,36,51,.74);
    border:1px solid rgba(125,211,252,.11);
}

.nir-disease .icon {
    font-size:25px;
    margin-bottom:7px;
}

.nir-disease b {
    display:block;
    color:#d9f8ff;
    font-size:12px;
}

.nir-disease span {
    display:block;
    color:#7399aa;
    font-size:9px;
    margin-top:4px;
}

/* ---------- Privacy banner ---------- */
.nir-privacy {
    margin:25px 0 20px;
    padding:22px 25px;
    display:flex;
    align-items:center;
    gap:17px;
    border-radius:22px;
    background:linear-gradient(100deg,rgba(16,185,129,.11),rgba(37,99,235,.10));
    border:1px solid rgba(94,234,212,.18);
}

.nir-privacy-icon {
    width:48px;
    height:48px;
    flex:0 0 48px;
    display:flex;
    align-items:center;
    justify-content:center;
    border-radius:15px;
    background:rgba(20,184,166,.14);
    font-size:23px;
}

.nir-privacy b {
    color:#dffffa;
    font-size:14px;
}

.nir-privacy span {
    display:block;
    color:#8eafbd;
    font-size:11px;
    line-height:1.5;
    margin-top:3px;
}

/* ---------- Footer ---------- */
.nir-footer {
    margin-top:10px;
    padding:24px 28px;
    border-radius:22px;
    background:linear-gradient(110deg,#071d2d,#0a2940);
    border:1px solid rgba(125,211,252,.12);
    color:#dff9ff;
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:20px;
}

.nir-footer b {
    font-size:17px;
}

.nir-footer small {
    display:block;
    color:#7fa5b7;
    font-size:10px;
    margin-top:4px;
}

.nir-footer-links {
    color:#8db2c1;
    font-size:11px;
}

.nir-footer-ai {
    color:#7fffea;
    font-size:12px;
    font-weight:800;
}

/* ---------- Responsive ---------- */
@media(max-width:1100px) {
    .nir-home-left { padding-left:38px; }
    .nir-home-title { font-size:44px !important; }
    .nir-stats { grid-template-columns:repeat(2,1fr); }
    .nir-disease-grid { grid-template-columns:repeat(3,1fr); }
}

@media(max-width:800px) {
    .nir-home-hero { display:block; }
    .nir-home-left, .nir-home-visual { width:100%; }
    .nir-home-left { padding:42px 28px 25px; }
    .nir-home-visual { padding:10px 22px 28px; }
    .nir-home-visual img { height:300px; }
    .nir-feature-grid { grid-template-columns:1fr; }
    .nir-stats { grid-template-columns:1fr 1fr; }
    .nir-footer { display:block; }
    .nir-footer-links { margin:12px 0; }
}

@media(max-width:520px) {
    .block-container { padding-left:1rem !important; padding-right:1rem !important; }
    .nir-home-title { font-size:37px !important; letter-spacing:-1.5px !important; }
    .nir-home-description { font-size:14px; }
    .nir-stats, .nir-disease-grid { grid-template-columns:1fr 1fr; }
    .nir-card-one, .nir-card-two, .nir-card-three { display:none; }
}
</style>
""", unsafe_allow_html=True)

# Self-contained hero image: no external URL or missing local asset required.
# The SVG is embedded directly into the page and therefore always renders in Streamlit.
NIRMAYA_HERO_SVG = r"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 700">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#08263a"/>
    <stop offset="0.55" stop-color="#07394a"/>
    <stop offset="1" stop-color="#161447"/>
  </linearGradient>
  <linearGradient id="pulse" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#4df5d5"/>
    <stop offset="0.5" stop-color="#39cfff"/>
    <stop offset="1" stop-color="#9b7cff"/>
  </linearGradient>
  <radialGradient id="glow">
    <stop offset="0" stop-color="#31e6d0" stop-opacity=".35"/>
    <stop offset="1" stop-color="#31e6d0" stop-opacity="0"/>
  </radialGradient>
  <filter id="shadow"><feDropShadow dx="0" dy="20" stdDeviation="22" flood-opacity=".35"/></filter>
  <filter id="soft"><feGaussianBlur stdDeviation="16"/></filter>
</defs>
<rect width="900" height="700" rx="42" fill="url(#bg)"/>
<circle cx="180" cy="130" r="220" fill="url(#glow)"/>
<circle cx="760" cy="590" r="250" fill="#6955ff" opacity=".13" filter="url(#soft)"/>
<!-- medical grid -->
<g opacity=".12" stroke="#9deeff">
  <path d="M70 120H830M70 210H830M70 300H830M70 390H830M70 480H830M70 570H830"/>
  <path d="M150 70V630M270 70V630M390 70V630M510 70V630M630 70V630M750 70V630"/>
</g>
<!-- central AI medical core -->
<g filter="url(#shadow)">
  <circle cx="450" cy="330" r="172" fill="#071b2a" stroke="#42e8d1" stroke-opacity=".45" stroke-width="2"/>
  <circle cx="450" cy="330" r="138" fill="#0b3042" stroke="#5bcfff" stroke-opacity=".28"/>
  <circle cx="450" cy="330" r="104" fill="#071b2a" stroke="url(#pulse)" stroke-width="4"/>
</g>
<!-- shield -->
<path d="M450 238 L520 264 V329 C520 391 486 432 450 451 C414 432 380 391 380 329 V264 Z" fill="url(#pulse)" opacity=".95"/>
<path d="M450 258 L501 278 V326 C501 370 478 399 450 416 C422 399 399 370 399 326 V278 Z" fill="#071b2a"/>
<path d="M450 285V371M407 328H493" stroke="#eaffff" stroke-width="12" stroke-linecap="round"/>
<!-- ECG -->
<path d="M285 330 H330 L350 330 L365 300 L382 360 L400 330 H430" fill="none" stroke="#5df5dc" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M470 330 H500 L516 330 L532 298 L550 360 L568 330 H615" fill="none" stroke="#58d9ff" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
<!-- orbit nodes -->
<g fill="#102f42" stroke="#54e9d5" stroke-width="2">
  <circle cx="450" cy="112" r="24"/><circle cx="675" cy="330" r="24"/><circle cx="450" cy="548" r="24"/><circle cx="225" cy="330" r="24"/>
</g>
<g fill="#dffeff" font-family="Arial, sans-serif" font-size="23" text-anchor="middle">
  <text x="450" y="120">AI</text><text x="675" y="338">+</text><text x="450" y="556">♥</text><text x="225" y="338">✚</text>
</g>
<!-- floating cards -->
<g font-family="Arial, sans-serif" filter="url(#shadow)">
  <rect x="80" y="88" width="205" height="76" rx="18" fill="#071b2a" fill-opacity=".92" stroke="#61e9da" stroke-opacity=".25"/>
  <text x="103" y="119" fill="#7fffea" font-size="14" font-weight="700">AI SCREENING</text>
  <text x="103" y="143" fill="#9bbaca" font-size="11">Smart health insights</text>
  <rect x="615" y="86" width="205" height="76" rx="18" fill="#071b2a" fill-opacity=".92" stroke="#78c9ff" stroke-opacity=".25"/>
  <text x="638" y="117" fill="#76dcff" font-size="14" font-weight="700">REPORT OCR</text>
  <text x="638" y="141" fill="#9bbaca" font-size="11">Extract medical values</text>
  <rect x="650" y="480" width="170" height="76" rx="18" fill="#071b2a" fill-opacity=".92" stroke="#a78bfa" stroke-opacity=".28"/>
  <text x="673" y="511" fill="#b8a4ff" font-size="14" font-weight="700">PRIVACY FIRST</text>
  <text x="673" y="535" fill="#9bbaca" font-size="11">Patient-focused records</text>
</g>
<!-- bottom label -->
<text x="450" y="650" text-anchor="middle" fill="#dffaff" font-family="Arial, sans-serif" font-size="18" font-weight="700" letter-spacing="3">NIRMAYA AI • HEALTH INTELLIGENCE</text>
</svg>"""
NIRMAYA_HERO_DATA = "data:image/svg+xml;base64," + base64.b64encode(NIRMAYA_HERO_SVG.encode("utf-8")).decode("ascii")

if page == "Home":
    # Keep the HTML as one continuous block. Blank lines inside nested <div>
    # elements can make Streamlit's Markdown parser display the inner HTML as code.
    st.markdown(f"""
<div class="nir-home-hero"><div class="nir-home-left"><div class="nir-ai-badge"><span class="nir-ai-dot"></span>NIRMAYA AI • INTELLIGENT HEALTH SCREENING</div><h1 class="nir-home-title">Your Health.<br><span class="mint">Our Intelligence.</span><br><span class="violet">A Healthier Tomorrow.</span></h1><p class="nir-home-description">NIRMAYA AI brings machine learning, medical-report OCR and intelligent health guidance together in one modern screening platform — designed to help you understand your health earlier.</p><div class="nir-hero-pills"><span class="nir-hero-pill">🤖 AI-Assisted Screening</span><span class="nir-hero-pill">🧠 Multi-Disease Models</span><span class="nir-hero-pill">📄 Report OCR</span><span class="nir-hero-pill">🔐 Patient Records</span></div></div><div class="nir-home-visual"><img src="{NIRMAYA_HERO_DATA}" alt="NIRMAYA AI healthcare visualization"><div class="nir-orbit-card nir-card-one"><div class="nir-orbit-icon">🧠</div><div><b>AI Screening</b><small>Data-driven health insights</small></div></div><div class="nir-orbit-card nir-card-two"><div class="nir-orbit-icon">🛡️</div><div><b>Privacy First</b><small>Patient-focused records</small></div></div><div class="nir-orbit-card nir-card-three"><div class="nir-orbit-icon">⚡</div><div><b>Smart &amp; Fast</b><small>Designed for easy screening</small></div></div></div></div>
""", unsafe_allow_html=True)
    st.markdown('<div class="nir-start-wrap">', unsafe_allow_html=True)
    if st.button("🚀  Start Your Health Journey  →", type="primary", key="home_start_health_check"):
        st.session_state["page"] = "Prediction"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="nir-stats"><div class="nir-stat"><div class="nir-stat-icon">🧬</div><b>Multi-Disease Screening</b><span>Independent ML models for different health conditions.</span></div><div class="nir-stat"><div class="nir-stat-icon">📊</div><b>Explainable Results</b><span>Clear screening indications and supporting information.</span></div><div class="nir-stat"><div class="nir-stat-icon">📄</div><b>Medical Report OCR</b><span>Extract useful values from uploaded medical reports.</span></div><div class="nir-stat"><div class="nir-stat-icon">👤</div><b>Patient Portal</b><span>Secure access to your own screening history.</span></div></div><div class="nir-feature-heading">Why NIRMAYA AI?</div><div class="nir-feature-subtitle">Technology designed around early screening, clarity and better health decisions.</div><div class="nir-feature-grid"><div class="nir-feature-card"><div class="nir-feature-icon">🤖</div><h3>Intelligent Screening</h3><p>Machine-learning models transform clinical inputs into easy-to-understand screening indications.</p></div><div class="nir-feature-card"><div class="nir-feature-icon">📋</div><h3>Report-to-Insight</h3><p>OCR support can extract relevant information from medical reports and reduce repetitive manual entry.</p></div><div class="nir-feature-card"><div class="nir-feature-icon">🩺</div><h3>Doctor + Patient Workflow</h3><p>Separate patient and doctor portals keep the screening workflow organized and practical.</p></div></div><div class="nir-disease-section"><div class="nir-feature-heading">One Platform. Multiple Health Checks.</div><div class="nir-feature-subtitle">Disease-specific models available in your NIRMAYA screening workflow.</div><div class="nir-disease-grid"><div class="nir-disease"><div class="icon">❤️</div><b>Heart Disease</b><span>Cardiovascular</span></div><div class="nir-disease"><div class="icon">🧠</div><b>Parkinson's</b><span>Neurological</span></div><div class="nir-disease"><div class="icon">🩸</div><b>Diabetes</b><span>Metabolic</span></div><div class="nir-disease"><div class="icon">🫘</div><b>Kidney Disease</b><span>Renal health</span></div><div class="nir-disease"><div class="icon">❤️</div><b>Heart Failure</b><span>Cardiac health</span></div><div class="nir-disease"><div class="icon">🎗️</div><b>Breast Cancer</b><span>Oncology</span></div></div></div><div class="nir-privacy"><div class="nir-privacy-icon">🛡️</div><div><b>Your health information deserves care.</b><span>NIRMAYA is designed as a screening-support and educational platform. AI outputs are not medical diagnoses and should be reviewed with a qualified healthcare professional.</span></div></div><footer class="nir-footer"><div><b>🤖 NIRMAYA AI</b><small>Early Detection • Better Decisions • Healthier Life</small></div><div class="nir-footer-links">Home &nbsp; • &nbsp; About Us &nbsp; • &nbsp; How It Works &nbsp; • &nbsp; Risk Checks</div><div class="nir-footer-ai">AI-POWERED HEALTH SCREENING</div></footer>
""", unsafe_allow_html=True)

# ============================================================
# ============================================================
# 💜 NIRMAYA AI — FINAL AI PURPLE LIGHT PREMIUM HEALTHCARE THEME
# ============================================================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp { background:#F7F5FF !important; color:#241B3A !important; }
.stApp { background: radial-gradient(circle at 5% 5%,rgba(124,77,255,.08),transparent 25%), radial-gradient(circle at 95% 8%,rgba(167,139,250,.12),transparent 28%), linear-gradient(135deg,#FCFBFF 0%,#F5F1FF 55%,#FAF8FF 100%) !important; }
.block-container { max-width:1536px !important; padding-top:.5rem !important; }
h1,h2,h3,h4,h5,h6 { color:#241B3A !important; } p,label,.stMarkdown,.stCaption,span { color:#655B78 !important; }
.nir-brand { color:#241B3A !important; } .nir-name { background:linear-gradient(90deg,#6D28D9,#8B5CF6,#A855F7)!important;-webkit-background-clip:text!important;background-clip:text!important;color:transparent!important; } .nir-tag{color:#8A7FA0!important}.nir-shield{background:linear-gradient(150deg,#7C3AED,#A855F7)!important;box-shadow:0 7px 22px rgba(124,58,237,.20)!important}
div[data-testid="stHorizontalBlock"] button { background:rgba(255,255,255,.82)!important;color:#5E5374!important;border:1px solid #E4DDF4!important;border-radius:12px!important;font-weight:700!important;min-height:46px!important;box-shadow:0 5px 18px rgba(80,50,130,.05)!important }
div[data-testid="stHorizontalBlock"] button:hover { background:#F1EAFF!important;color:#6D28D9!important;border-color:#BFA7F7!important;box-shadow:0 0 20px rgba(124,58,237,.10)!important }
.theme-dot{background:#F3EEFF!important;color:#7C3AED!important;border:1px solid #DDD1F7!important}
.nir-home-hero { background:linear-gradient(135deg,#FFFFFF 0%,#F5EEFF 52%,#F8F4FF 100%)!important;border:1px solid #E3D8F7!important;box-shadow:0 30px 80px rgba(83,54,130,.12),inset 0 1px 0 #fff!important }
.nir-home-hero:before{background:radial-gradient(circle,rgba(124,58,237,.12),transparent 68%)!important}.nir-home-hero:after{background:radial-gradient(circle,rgba(168,85,247,.12),transparent 68%)!important}
.nir-home-title{color:#241B3A!important}.nir-home-title .mint{background:linear-gradient(90deg,#6D28D9,#8B5CF6)!important;-webkit-background-clip:text!important;background-clip:text!important;color:transparent!important}.nir-home-title .violet{background:linear-gradient(90deg,#8B5CF6,#C026D3)!important;-webkit-background-clip:text!important;background-clip:text!important;color:transparent!important}.nir-home-description{color:#6D6380!important}
.nir-ai-badge{background:linear-gradient(90deg,#F1EAFF,#F8EEFF)!important;border-color:#D8C8F8!important;color:#6D28D9!important}.nir-ai-dot{background:#8B5CF6!important;box-shadow:0 0 14px #A855F7!important}.nir-hero-pill{background:rgba(255,255,255,.9)!important;border-color:#E3D8F7!important;color:#6D6380!important}.nir-hero-pill:hover{border-color:#9B74EA!important;color:#6D28D9!important;background:#F5EFFF!important}
.nir-stat,.nir-feature-card,.nir-disease,.nir-privacy,.nir-footer,.nir-orbit-card{background:linear-gradient(145deg,#FFFFFF,#F8F5FF)!important;border:1px solid #E4DCF3!important;box-shadow:0 15px 40px rgba(76,48,120,.08)!important}.nir-stat:hover,.nir-feature-card:hover,.nir-disease:hover{border-color:#C8B1F4!important;box-shadow:0 20px 45px rgba(76,48,120,.12),0 0 25px rgba(124,58,237,.06)!important}.nir-stat b,.nir-feature-heading,.nir-disease b{color:#2D2145!important}.nir-stat span,.nir-feature-subtitle,.nir-disease span{color:#766B88!important}
.nir-section-title,.nir-section-title-highlight{color:#2D2145!important}.nir-section-title-highlight{background:linear-gradient(90deg,#6D28D9,#A855F7)!important;-webkit-background-clip:text!important;background-clip:text!important;color:transparent!important}
.stTextInput input,.stTextArea textarea,.stNumberInput input,[data-baseweb="select"] > div,[data-baseweb="input"] > div{background:#FFFFFF!important;color:#2D2145!important;border:1px solid #DDD3EE!important;border-radius:13px!important}.stTextInput input:focus,.stTextArea textarea:focus,.stNumberInput input:focus{border-color:#9B74EA!important;box-shadow:0 0 0 2px rgba(124,58,237,.10)!important}[data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"]{background:#FFFFFF!important;color:#2D2145!important;border:1px solid #DDD3EE!important}[data-baseweb="option"]{color:#4D4261!important}[data-baseweb="option"]:hover{background:#F2EAFF!important;color:#6D28D9!important}
.stButton > button{background:linear-gradient(100deg,#6D28D9,#8B5CF6 58%,#A855F7)!important;color:#FFFFFF!important;border:none!important;border-radius:13px!important;font-weight:800!important;box-shadow:0 8px 25px rgba(109,40,217,.20)!important}.stButton > button:hover{transform:translateY(-2px);box-shadow:0 12px 35px rgba(109,40,217,.25)!important}
[data-testid="stFileUploader"]{background:#FFFFFF!important;border:1px dashed #BFA7F7!important;border-radius:18px!important}[data-testid="stFileUploaderDropzone"]{background:#FAF8FF!important}
.stRadio > div{background:#FFFFFF!important;border:1px solid #E0D7EF!important;border-radius:14px!important}
[data-testid="stMetric"]{background:linear-gradient(145deg,#FFFFFF,#F7F3FF)!important;border:1px solid #E1D8F0!important;border-radius:18px!important;box-shadow:0 12px 30px rgba(76,48,120,.07)!important}[data-testid="stDataFrame"]{border:1px solid #E1D8F0!important}
[data-testid="stTabs"] button{color:#756A88!important}[data-testid="stTabs"] button:hover{color:#6D28D9!important}[data-testid="stTabs"] button[aria-selected="true"]{color:#6D28D9!important}
.patient-message-card{background:linear-gradient(135deg,#FFFFFF,#F6F0FF)!important;border:1px solid #E0D4F3!important}.patient-message-card b{color:#6D28D9!important}.patient-message-card small{color:#8B7F9E!important}.patient-message-card p{color:#413552!important}.doctor-reply-card{background:linear-gradient(135deg,#F1EAFF,#F8EEFF)!important;border:1px solid #D5C2F5!important;color:#3E2D57!important}.doctor-reply-card span{color:#513A72!important}.doctor-reply-card small{color:#7D6E91!important}
.stAlert{background:#FAF8FF!important;border:1px solid #DDD1F1!important;border-radius:14px!important;color:#4A3D5D!important}.nirmaya-chat-box{background:linear-gradient(145deg,#FFFFFF,#F7F3FF)!important;border:1px solid #E1D8F0!important;box-shadow:0 15px 40px rgba(76,48,120,.10)!important}div[data-testid="stTextArea"] textarea{background:#FFFFFF!important;color:#2D2145!important;border:1px solid #DDD3EE!important}
.nir-footer{background:linear-gradient(145deg,#F3EEFF,#FFFFFF)!important;border:1px solid #E0D6F0!important;color:#665A79!important}.nir-footer small{color:#887C9A!important}.nir-footer-ai{color:#6D28D9!important}hr{border-color:#E1D8EF!important}::-webkit-scrollbar{width:8px}::-webkit-scrollbar-track{background:#F7F5FF}::-webkit-scrollbar-thumb{background:#CFC1E7;border-radius:10px}::-webkit-scrollbar-thumb:hover{background:#9B74EA!important}
@media(max-width:768px){[data-testid="stPopover"]{right:16px!important;bottom:16px!important}.nir-home-title{font-size:40px!important}}
</style>
""", unsafe_allow_html=True)

# MODEL PATHS
# ============================================================
# ============================================================

# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {

    "Heart Disease":
        os.path.join(BASE_DIR, "models", "heart_disease"),

    "Parkinson's Disease":
        os.path.join(BASE_DIR, "models", "parkinsons"),

    "Diabetes":
        os.path.join(BASE_DIR, "models", "diabetes"),

    "Chronic Kidney Disease":
        os.path.join(BASE_DIR, "models", "kidney"),

    "Heart Failure":
        os.path.join(BASE_DIR, "models", "heart_failure"),

    "Breast Cancer":
        os.path.join(BASE_DIR, "models", "breast_cancer")
}


# ============================================================
# MODEL ACCURACY
# ============================================================
#
# These values must be TEST-SET accuracies from your training.
#
# Heart Disease = confirmed from your completed training:
# 0.901639 = 90.16%
#
# Do NOT put prediction confidence here.
# ============================================================

MODEL_ACCURACY = {

    "Heart Disease": 90.16,

    # Add the actual test accuracy from your training output
    # when available.
    "Parkinson's Disease": None,

    "Diabetes": None,

    "Chronic Kidney Disease": None,

    "Heart Failure": None,

    "Breast Cancer": None
}


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    # ========================================================
    # HEART DISEASE
    # ========================================================

    "Heart Disease": {

        "icon": "❤️",

        "risk": [
            "High blood pressure",
            "High cholesterol",
            "Smoking",
            "Diabetes or high blood sugar",
            "Low physical activity",
            "Increasing age",
            "Family history",
            "Certain patterns of chest pain"
        ],

        "nutrients": [
            "Omega-3 fatty acids",
            "Dietary fiber",
            "Potassium",
            "Magnesium",
            "Vitamin D",
            "Folate"
        ],

        "precautions": [
            "Monitor blood pressure regularly",
            "Monitor cholesterol",
            "Maintain a balanced diet",
            "Exercise according to medical advice",
            "Avoid smoking",
            "Maintain a healthy weight",
            "Consult a healthcare professional"
        ]
    },


    # ========================================================
    # PARKINSON'S DISEASE
    # ========================================================

    "Parkinson's Disease": {

        "icon": "🧠",

        "risk": [
            "Increasing age",
            "Family history",
            "Certain genetic factors",
            "Environmental exposures",
            "Some occupational exposures"
        ],

        "nutrients": [
            "Vitamin D",
            "Vitamin B12",
            "Folate",
            "Magnesium",
            "Omega-3 fatty acids",
            "Calcium"
        ],

        "precautions": [
            "Maintain regular physical activity",
            "Maintain balanced nutrition",
            "Get adequate sleep",
            "Discuss new neurological symptoms with a doctor",
            "Attend regular medical follow-up"
        ]
    },


    # ========================================================
    # DIABETES
    # ========================================================

    "Diabetes": {

        "icon": "🩸",

        "risk": [
            "High blood glucose",
            "Overweight or obesity",
            "Low physical activity",
            "Family history",
            "High blood pressure",
            "Unhealthy dietary patterns",
            "Increasing age"
        ],

        "nutrients": [
            "Dietary fiber",
            "Vitamin D",
            "Magnesium",
            "Vitamin B12",
            "Chromium",
            "Folate"
        ],

        "precautions": [
            "Monitor blood glucose",
            "Maintain a balanced diet",
            "Exercise regularly when appropriate",
            "Maintain a healthy weight",
            "Limit excess added sugar",
            "Follow medical advice"
        ]
    },


    # ========================================================
    # CHRONIC KIDNEY DISEASE
    # ========================================================

    "Chronic Kidney Disease": {

        "icon": "🫘",

        "risk": [
            "Diabetes",
            "High blood pressure",
            "Kidney disease family history",
            "Cardiovascular disease",
            "Smoking",
            "Older age"
        ],

        "nutrients": [
            "Iron",
            "Vitamin B12",
            "Folate",
            "Vitamin D",
            "Calcium"
        ],

        "precautions": [
            "Monitor blood pressure",
            "Monitor blood glucose if applicable",
            "Take medicines only as prescribed",
            "Follow an appropriate diet recommended by a healthcare professional",
            "Attend kidney-function follow-ups"
        ]
    },


    # ========================================================
    # HEART FAILURE
    # ========================================================

    "Heart Failure": {

        "icon": "❤️",

        "risk": [
            "High blood pressure",
            "Coronary artery disease",
            "Previous heart attack",
            "Diabetes",
            "Obesity",
            "Smoking",
            "Kidney disease"
        ],

        "nutrients": [
            "Vitamin D",
            "Magnesium",
            "Potassium",
            "Thiamine",
            "Iron",
            "Folate"
        ],

        "precautions": [
            "Monitor blood pressure",
            "Follow prescribed medication",
            "Monitor weight changes",
            "Follow appropriate dietary advice",
            "Avoid smoking",
            "Attend regular medical follow-up"
        ]
    },


    # ========================================================
    # BREAST CANCER
    # ========================================================

    "Breast Cancer": {

        "icon": "🎗️",

        "risk": [
            "Increasing age",
            "Family history",
            "Certain genetic factors",
            "Hormonal and reproductive factors",
            "Alcohol exposure",
            "Obesity after menopause",
            "Previous radiation exposure"
        ],

        "nutrients": [
            "Vitamin D",
            "Folate",
            "Vitamin B12",
            "Calcium",
            "Vitamin C",
            "Vitamin E"
        ],

        "precautions": [
            "Follow recommended screening",
            "Discuss unusual breast changes with a doctor",
            "Maintain a healthy weight",
            "Stay physically active",
            "Avoid smoking",
            "Follow professional medical advice"
        ]
    }
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(disease):

    if disease not in MODEL_PATHS:
        raise ValueError(
            f"Unknown disease: {disease}"
        )

    folder = MODEL_PATHS[disease]

    model_path = os.path.join(
        folder,
        "best_model.pkl"
    )

    scaler_path = os.path.join(
        folder,
        "scaler.pkl"
    )

    features_path = os.path.join(
        folder,
        "features.pkl"
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    for path, label in [
        (model_path, "model"),
        (scaler_path, "scaler"),
        (features_path, "features")
    ]:

        if not os.path.isfile(path):

            raise FileNotFoundError(
                f"Missing {label}: {path}"
            )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    model = joblib.load(model_path)

    scaler = joblib.load(scaler_path)

    features = list(
        joblib.load(features_path)
    )

    # --------------------------------------------------------
    # CHECK MODEL FIT
    # --------------------------------------------------------

    try:

        check_is_fitted(model)

    except Exception as e:

        raise ValueError(
            f"The saved model for {disease} is not fitted.\n"
            f"File: {model_path}"
        ) from e

    # --------------------------------------------------------
    # CHECK SCALER FIT
    # --------------------------------------------------------

    try:

        check_is_fitted(scaler)

    except Exception as e:

        raise ValueError(
            f"The saved scaler for {disease} is not fitted.\n"
            f"File: {scaler_path}"
        ) from e

    # --------------------------------------------------------
    # CHECK FEATURES
    # --------------------------------------------------------

    if len(features) == 0:

        raise ValueError(
            f"No features found for {disease}."
        )

    # --------------------------------------------------------
    # CHECK SCALER FEATURE COUNT
    # --------------------------------------------------------

    scaler_features = getattr(
        scaler,
        "n_features_in_",
        None
    )

    if (
        scaler_features is not None
        and scaler_features != len(features)
    ):

        raise ValueError(
            f"Feature mismatch for {disease}. "
            f"Scaler expects {scaler_features} features "
            f"but features.pkl contains {len(features)}."
        )

    return model, scaler, features


# ============================================================
# SAFE STREAMLIT KEY
# ============================================================

def widget_key(prefix, index, feature):

    clean_feature = (
        str(feature)
        .replace(" ", "_")
        .replace(".", "_")
        .replace(":", "_")
        .replace("/", "_")
    )

    return (
        f"{prefix}_{index}_{clean_feature}"
    )


# ============================================================
# OCR PREFILL HELPER
# ============================================================

def _normalise_ocr_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _ocr_number(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def apply_ocr_to_inputs(disease, features, ocr_data, prefix):
    """Safely prefill Streamlit input state from OCR values.

    Only clear feature matches are applied. Missing/ambiguous values are
    left for manual entry; the app never invents a value.
    """
    if not isinstance(ocr_data, dict):
        return

    # OCR responses can contain nested dictionaries such as
    # {"medical_values": {"Cholesterol": {"result": "236"}}}.
    # Flatten them so matching can use the actual field names without
    # ever showing the raw OCR/JSON response to the user.
    source = {}

    def collect_values(obj, parent_key=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{parent_key} {key}".strip()
                if isinstance(value, dict):
                    collect_values(value, full_key)
                else:
                    source[_normalise_ocr_key(full_key)] = value
                    source[_normalise_ocr_key(key)] = value
        elif parent_key:
            source[_normalise_ocr_key(parent_key)] = obj

    collect_values(ocr_data)

    aliases = {
        "age": ["age", "patientage"],
        "sex": ["sex", "gender"],
        "cp": ["cp", "chestpaintype", "chestpain"],
        "trestbps": ["trestbps", "restingbloodpressure", "bloodpressure", "bp"],
        "chol": ["chol", "cholesterol", "totalcholesterol"],
        "fbs": ["fbs", "fastingbloodsugar", "fastingglucose", "glucose"],
        "restecg": ["restecg", "restingecg", "ecg"],
        "thalach": ["thalach", "maximumheartrate", "maxheartrate", "heartrate"],
        "exang": ["exang", "exerciseinducedangina", "angina"],
        "oldpeak": ["oldpeak", "stdepression", "stdepressionoldpeak"],
        "slope": ["slope"],
        "ca": ["ca", "numberofmajorvessels", "majorvessels"],
        "thal": ["thal"]
    }

    def find_value(names):
        for name in names:
            key = _normalise_ocr_key(name)
            if key in source:
                return source[key]
        return None

    if disease == "Heart Disease":
        mapping = {
            "age": find_value(aliases["age"]),
            "sex": find_value(aliases["sex"]),
            "cp": find_value(aliases["cp"]),
            "bp": find_value(aliases["trestbps"]),
            "chol": find_value(aliases["chol"]),
            "fbs": find_value(aliases["fbs"]),
            "restecg": find_value(aliases["restecg"]),
            "thalach": find_value(aliases["thalach"]),
            "exang": find_value(aliases["exang"]),
            "oldpeak": find_value(aliases["oldpeak"]),
            "slope": find_value(aliases["slope"]),
            "ca": find_value(aliases["ca"]),
            "thal": find_value(aliases["thal"]),
        }

        for field, raw in mapping.items():
            if raw is None:
                continue

            if field == "sex":
                text = str(raw).strip().lower()
                if text in ("male", "m", "man"):
                    st.session_state[f"{prefix}_heart_sex"] = "Male"
                elif text in ("female", "f", "woman"):
                    st.session_state[f"{prefix}_heart_sex"] = "Female"
                continue

            number = _ocr_number(raw)
            if number is None:
                continue

            state_keys = {
                "age": f"{prefix}_heart_age",
                "cp": f"{prefix}_heart_cp",
                "bp": f"{prefix}_heart_bp",
                "chol": f"{prefix}_heart_chol",
                "fbs": f"{prefix}_heart_fbs",
                "restecg": f"{prefix}_heart_restecg",
                "thalach": f"{prefix}_heart_thalach",
                "exang": f"{prefix}_heart_exang",
                "oldpeak": f"{prefix}_heart_oldpeak",
                "slope": f"{prefix}_heart_slope",
                "ca": f"{prefix}_heart_ca",
                "thal": f"{prefix}_heart_thal",
            }
            key = state_keys[field]

            # Keep values inside the actual widget choices/ranges.
            if field in ("cp", "restecg", "slope", "thal") and number not in [0, 1, 2, 3]:
                continue
            if field in ("fbs", "exang") and number not in [0, 1]:
                continue
            if field == "ca" and not 0 <= number <= 4:
                continue
            if field == "age" and not 1 <= number <= 120:
                continue
            if field == "bp" and not 50 <= number <= 300:
                continue
            if field == "chol" and not 50 <= number <= 700:
                continue
            if field == "thalach" and not 30 <= number <= 300:
                continue
            if field == "oldpeak" and not 0 <= number <= 10:
                continue

            st.session_state[key] = int(number) if field != "oldpeak" else float(number)

    else:
        # Generic numeric prefill for diseases whose feature names match
        # OCR keys. Categorical fields remain manual unless safely matched.
        for i, feature in enumerate(features):
            feature_key = _normalise_ocr_key(feature)
            raw = source.get(feature_key)
            if raw is None:
                continue
            number = _ocr_number(raw)
            if number is not None:
                st.session_state[widget_key(prefix, i, feature)] = number

# ============================================================
# DISEASE INPUT FORM
# ============================================================

def create_disease_inputs(
    disease,
    features,
    prefix
):

    values = []


    # ========================================================
    # HEART DISEASE
    # ========================================================

    if disease == "Heart Disease":

        st.subheader(
            "❤️ Heart Disease Medical Information"
        )

        c1, c2, c3 = st.columns(3)

        # ----------------------------------------------------
        # COLUMN 1
        # ----------------------------------------------------

        with c1:

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=40,
                key=f"{prefix}_heart_age"
            )

            sex = st.selectbox(
                "Sex",
                ["Male", "Female"],
                key=f"{prefix}_heart_sex"
            )

            cp = st.selectbox(
                "Chest Pain Type",
                [0, 1, 2, 3],
                key=f"{prefix}_heart_cp"
            )

            trestbps = st.number_input(
                "Resting Blood Pressure",
                min_value=50,
                max_value=300,
                value=120,
                key=f"{prefix}_heart_bp"
            )

        # ----------------------------------------------------
        # COLUMN 2
        # ----------------------------------------------------

        with c2:

            chol = st.number_input(
                "Cholesterol",
                min_value=50,
                max_value=700,
                value=200,
                key=f"{prefix}_heart_chol"
            )

            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                [0, 1],
                key=f"{prefix}_heart_fbs"
            )

            restecg = st.selectbox(
                "Resting ECG",
                [0, 1, 2],
                key=f"{prefix}_heart_restecg"
            )

            thalach = st.number_input(
                "Maximum Heart Rate",
                min_value=30,
                max_value=300,
                value=150,
                key=f"{prefix}_heart_thalach"
            )

        # ----------------------------------------------------
        # COLUMN 3
        # ----------------------------------------------------

        with c3:

            exang = st.selectbox(
                "Exercise Induced Angina",
                [0, 1],
                key=f"{prefix}_heart_exang"
            )

            oldpeak = st.number_input(
                "ST Depression (Oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key=f"{prefix}_heart_oldpeak"
            )

            slope = st.selectbox(
                "Slope",
                [0, 1, 2],
                key=f"{prefix}_heart_slope"
            )

            ca = st.number_input(
                "Number of Major Vessels (CA)",
                min_value=0,
                max_value=4,
                value=0,
                key=f"{prefix}_heart_ca"
            )

            thal = st.selectbox(
                "Thal",
                [0, 1, 2, 3],
                key=f"{prefix}_heart_thal"
            )

        sex_value = (
            1
            if sex == "Male"
            else 0
        )

        values = [
            age,
            sex_value,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]


    # ========================================================
    # PARKINSON'S
    # ========================================================

    elif disease == "Parkinson's Disease":

        st.subheader(
            "🧠 Parkinson's Disease Medical Information"
        )

        st.info(
            "Enter the voice-analysis measurements "
            "used by the trained Parkinson's model."
        )

        columns = st.columns(3)

        for i, feature in enumerate(features):

            with columns[i % 3]:

                value = st.number_input(
                    str(feature),
                    value=0.0,
                    format="%.8f",
                    key=widget_key(
                        prefix,
                        i,
                        feature
                    )
                )

                values.append(value)


    # ========================================================
    # DIABETES
    # ========================================================

    elif disease == "Diabetes":

        st.subheader(
            "🩸 Diabetes Medical Information"
        )

        st.info(
            "Enter the clinical information used "
            "by the diabetes model."
        )

        columns = st.columns(3)

        for i, feature in enumerate(features):

            name = (
                str(feature)
                .strip()
                .lower()
            )

            with columns[i % 3]:

                # AGE
                if name == "age":

                    value = st.number_input(
                        "Age",
                        min_value=1,
                        max_value=120,
                        value=40,
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                # GENDER
                elif (
                    "gender" in name
                    or name == "sex"
                ):

                    selected = st.selectbox(
                        "Gender",
                        ["Male", "Female"],
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                    value = (
                        1
                        if selected == "Male"
                        else 0
                    )

                # OTHER BINARY FEATURES
                else:

                    selected = st.selectbox(
                        str(feature),
                        ["No", "Yes"],
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                    value = (
                        1
                        if selected == "Yes"
                        else 0
                    )

                values.append(value)


    # ========================================================
    # CHRONIC KIDNEY DISEASE
    # ========================================================

    elif disease == "Chronic Kidney Disease":

        st.subheader(
            "🫘 Chronic Kidney Disease Information"
        )

        st.info(
            "Enter the laboratory and clinical "
            "measurements used by the kidney model."
        )

        categorical = [
            "rbc",
            "pc",
            "pcc",
            "ba",
            "htn",
            "dm",
            "cad",
            "appet",
            "pe",
            "ane"
        ]

        columns = st.columns(3)

        for i, feature in enumerate(features):

            name = (
                str(feature)
                .strip()
                .lower()
            )

            with columns[i % 3]:

                # AGE
                if name == "age":

                    value = st.number_input(
                        "Age",
                        min_value=1,
                        max_value=120,
                        value=45,
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                # NORMAL / ABNORMAL
                elif name in [
                    "rbc",
                    "pc"
                ]:

                    selected = st.selectbox(
                        str(feature),
                        [
                            "normal",
                            "abnormal"
                        ],
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                    value = (
                        1
                        if selected == "normal"
                        else 0
                    )

                # PRESENT / NOT PRESENT
                elif name in [
                    "pcc",
                    "ba"
                ]:

                    selected = st.selectbox(
                        str(feature),
                        [
                            "notpresent",
                            "present"
                        ],
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                    value = (
                        1
                        if selected == "present"
                        else 0
                    )

                # YES / NO
                elif name in [
                    "htn",
                    "dm",
                    "cad",
                    "pe",
                    "ane"
                ]:

                    selected = st.selectbox(
                        str(feature),
                        [
                            "no",
                            "yes"
                        ],
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                    value = (
                        1
                        if selected == "yes"
                        else 0
                    )

                # APPETITE
                elif name == "appet":

                    selected = st.selectbox(
                        "Appetite",
                        [
                            "good",
                            "poor"
                        ],
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                    value = (
                        1
                        if selected == "good"
                        else 0
                    )

                # NUMERIC
                else:

                    value = st.number_input(
                        str(feature),
                        value=0.0,
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                values.append(value)


    # ========================================================
    # HEART FAILURE
    # ========================================================

    elif disease == "Heart Failure":

        st.subheader(
            "❤️ Heart Failure Medical Information"
        )

        st.info(
            "Enter the clinical measurements "
            "used by the heart-failure model."
        )

        columns = st.columns(3)

        for i, feature in enumerate(features):

            name = (
                str(feature)
                .strip()
                .lower()
            )

            with columns[i % 3]:

                # AGE
                if name == "age":

                    value = st.number_input(
                        "Age",
                        min_value=1,
                        max_value=120,
                        value=50,
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                # BINARY FEATURES
                elif (
                    "anaemia" in name
                    or "diabetes" in name
                    or "high_blood_pressure" in name
                    or "smoking" in name
                    or name == "sex"
                ):

                    value = st.selectbox(
                        str(feature),
                        [0, 1],
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                # NUMERIC FEATURES
                else:

                    value = st.number_input(
                        str(feature),
                        value=0.0,
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                values.append(value)


    # ========================================================
    # BREAST CANCER
    # ========================================================

    elif disease == "Breast Cancer":

        st.subheader(
            "🎗️ Breast Cancer Medical Information"
        )

        st.info(
            "Enter the numerical measurements "
            "used by the breast-cancer model."
        )

        columns = st.columns(3)

        for i, feature in enumerate(features):

            with columns[i % 3]:

                value = st.number_input(
                    str(feature),
                    value=0.0,
                    format="%.6f",
                    key=widget_key(
                        prefix,
                        i,
                        feature
                    )
                )

                values.append(value)


    # ========================================================
    # RETURN
    # ========================================================

    return values


# ============================================================
# PREDICTION
# ============================================================

def make_prediction(
    model,
    scaler,
    features,
    values
):

    # --------------------------------------------------------
    # FEATURE COUNT
    # --------------------------------------------------------

    if len(values) != len(features):

        raise ValueError(
            f"Feature mismatch. "
            f"Model expects {len(features)} values "
            f"but received {len(values)}."
        )

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    try:

        check_is_fitted(model)

    except Exception as e:

        raise ValueError(
            "The loaded model is not fitted."
        ) from e

    # --------------------------------------------------------
    # CHECK SCALER
    # --------------------------------------------------------

    try:

        check_is_fitted(scaler)

    except Exception as e:

        raise ValueError(
            "The loaded scaler is not fitted."
        ) from e

    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------
       
    input_df = pd.DataFrame([values], columns=features)

    input_df = input_df.apply(pd.to_numeric, errors="coerce")
   
    # --------------------------------------------------------
    # CONVERT TO NUMERIC
    # --------------------------------------------------------

    input_df = input_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # CHECK INVALID VALUES
    # --------------------------------------------------------

    if input_df.isnull().any().any():

        bad_columns = (
            input_df.columns[
                input_df.isnull().any()
            ].tolist()
        )

        raise ValueError(
            f"Invalid values detected in: "
            f"{bad_columns}"
        )

    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    input_scaled = scaler.transform(
        input_df
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    prediction = model.predict(
        input_scaled
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    probability = None

    if hasattr(model, "predict_proba"):

        try:

            probabilities = (
                model.predict_proba(
                    input_scaled
                )
            )

            probability = float(
                np.max(
                    probabilities[0]
                ) * 100
            )

        except Exception:

            probability = None

    return (
        int(prediction[0]),
        probability
    )


# ============================================================
# MEDICAL REPORT HELPERS
# ============================================================

REPORT_DIR = os.path.join(BASE_DIR, "patient_reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def _safe_report_filename(filename):
    name = os.path.basename(filename or "medical_report")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:120] or "medical_report"


def extract_text_from_pdf(pdf_bytes):
    """Extract text from normal PDFs and OCR scanned PDF pages when possible."""
    text_parts = []

    if PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        except Exception:
            pass

    text = "\n".join(text_parts).strip()
    if text:
        return text

    # Scanned PDF fallback: render pages and send them through the existing OCR reader.
    if fitz is not None:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            ocr_parts = []
            for page_no, page in enumerate(doc):
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                temp_path = os.path.join(BASE_DIR, f"temp_pdf_page_{page_no}.png")
                pix.save(temp_path)
                try:
                    ocr_parts.append(extract_text_from_image(temp_path) or "")
                finally:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
            doc.close()
            return "\n".join(ocr_parts).strip()
        except Exception:
            return ""

    return ""


def save_uploaded_patient_report(uploaded_file, patient_id, patient_name):
    """Save an uploaded patient report and its extracted text."""
    if uploaded_file is None:
        return None

    original_name = _safe_report_filename(uploaded_file.name)
    extension = Path(original_name).suffix.lower()
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    if extension not in allowed:
        raise ValueError("Only PDF, PNG, JPG, JPEG and WEBP reports are supported.")

    patient_folder = os.path.join(REPORT_DIR, patient_id.upper())
    os.makedirs(patient_folder, exist_ok=True)

    # Add a short hash so repeated uploads with the same filename do not overwrite.
    data = uploaded_file.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:10]
    stored_name = f"{Path(original_name).stem}_{digest}{extension}"
    stored_path = os.path.join(patient_folder, stored_name)

    with open(stored_path, "wb") as f:
        f.write(data)

    extracted_text = ""
    if extension == ".pdf":
        extracted_text = extract_text_from_pdf(data)
    else:
        temp_path = stored_path
        try:
            extracted_text = extract_text_from_image(temp_path) or ""
        except Exception:
            extracted_text = ""

    report_id = save_patient_report(
        patient_id=patient_id,
        patient_name=patient_name,
        file_name=original_name,
        file_type=uploaded_file.type or extension.lstrip("."),
        file_path=stored_path,
        extracted_text=extracted_text
    )
    return {
        "id": report_id,
        "file_name": original_name,
        "file_path": stored_path,
        "extracted_text": extracted_text
    }


def show_patient_reports(patient_id, heading="📄 Medical Reports"):
    reports = get_patient_reports(patient_id)
    st.subheader(heading)
    if not reports:
        st.info("No medical reports have been uploaded for this patient.")
        return

    for report in reports:
        report_id, _, patient_name, file_name, file_type, file_path, extracted_text, uploaded_at = report
        with st.expander(f"📄 {file_name} · {uploaded_at}"):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.caption(f"Uploaded for: {patient_name} · Type: {file_type or 'document'}")
            with c2:
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download",
                            data=f.read(),
                            file_name=file_name,
                            mime=file_type or "application/octet-stream",
                            key=f"download_report_{report_id}"
                        )
            if extracted_text:
                st.markdown("**🤖 Extracted report text**")
                st.text_area(
                    "Report text",
                    extracted_text,
                    height=180,
                    key=f"report_text_{report_id}",
                    disabled=True
                )
            else:
                st.warning("No readable text was extracted. The original report is still available for download.")


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    disease,
    prediction,
    probability,
    user_type,
    patient_name=None,
    patient_gender=None,
    patient_age=None,
    input_data=None,
    patient_id=None,
    doctor_id=None
):

    info = DISEASE_INFO.get(
        disease,
        {}
    )

    # --------------------------------------------------------
    # SAFE FALLBACKS
    # --------------------------------------------------------

    risk_items = (
        info.get("risk")
        or [
            "Risk may depend on age, family history, lifestyle, medical history, and other clinical factors."
        ]
    )

    nutrient_items = (
        info.get("nutrients")
        or [
            "No disease-specific nutrient information has been configured."
        ]
    )

    precaution_items = (
        info.get("precautions")
        or [
            "Follow appropriate medical advice and recommended follow-up."
        ]
    )

    # --------------------------------------------------------
    # RESULT HEADER
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🔍 AI Screening Result"
    )
    
    # --------------------------------------------------------
    # RISK RESULT
    # --------------------------------------------------------
    if prediction == 1:

        st.error(
            f"⚠️ Higher Risk Indicated for {disease}"
        )

    else:

        st.success(
            f"✅ Lower Risk Indicated for {disease}"
        )
    # --------------------------------------------------------
    # ACCURACY + CONFIDENCE
    # --------------------------------------------------------
        # --------------------------------------------------------
    # 💾 SAVE PATIENT SCREENING RECORD
    # --------------------------------------------------------

    if user_type == "Patient":
        if patient_name:
            if st.button(
                "💾 Save Screening Record",
                key="save_patient_screening_record"
            ):
                patient_id = patient_id or get_or_create_patient_id(patient_name)
                save_patient_record(
                    patient_name=patient_name,
                    gender=patient_gender,
                    age=patient_age,
                    disease=disease,
                    prediction=("Higher Risk" if prediction == 1 else "Lower Risk"),
                    confidence=probability,
                    input_data=str(input_data if input_data is not None else {}),
                    patient_id=patient_id,
                    doctor_id=doctor_id
                )
                st.success("✅ Screening record saved successfully!")
                st.info(f"🆔 NIRAMAYA Patient ID: **{patient_id}**")

    elif user_type == "Doctor" and patient_id:
        save_patient_record(
            patient_name=patient_name,
            gender=patient_gender,
            age=patient_age,
            disease=disease,
            prediction=("Higher Risk" if prediction == 1 else "Lower Risk"),
            confidence=probability,
            input_data=str(input_data if input_data is not None else {}),
            patient_id=patient_id,
            doctor_id=doctor_id
        )
        st.success("✅ Doctor screening result saved to the patient's history.")

    col1, col2 = st.columns(2)

    # MODEL ACCURACY
    with col1:

        accuracy = MODEL_ACCURACY.get(
            disease
        )

        if accuracy is not None:

            st.metric(
                "Model Accuracy",
                f"{accuracy:.2f}%"
            )

        else:

            st.metric(
                "Model Accuracy",
                "Not stored"
            )

    # PREDICTION CONFIDENCE
    with col2:

        if probability is not None:

            st.metric(
                "Prediction Confidence",
                f"{probability:.2f}%"
            )

        else:

            st.metric(
                "Prediction Confidence",
                "N/A"
            )

    st.caption(
        "Model Accuracy represents performance measured during model evaluation. "
        "Prediction Confidence represents the model's probability for this individual input. "
        "They are different measures."
    )

    # --------------------------------------------------------
    # MEDICAL DISCLAIMER
    # --------------------------------------------------------

    st.warning(
        "This is an AI-based screening result, not a medical diagnosis. "
        "Please consult a qualified healthcare professional for clinical evaluation."
    )

    # ========================================================
    # WHY RISK MAY INCREASE
    # ========================================================

    st.subheader(
        "📌 Why Risk May Increase"
    )

    st.caption(
        "These are general factors associated with the condition. "
        "They are not a direct explanation of why this individual received this prediction."
    )

    for item in risk_items:

        st.write(
            f"• {item}"
        )

    # ========================================================
    # NUTRIENTS / DEFICIENCIES
    # ========================================================

    st.subheader(
        "🥗 Nutrients / Deficiencies That May Be Associated"
    )

    st.caption(
        "Nutrient deficiencies or dietary factors may be associated "
        "with health conditions, but they should not be assumed to "
        "be the cause of an individual's prediction."
    )

    for item in nutrient_items:

        st.write(
            f"• {item}"
        )

    # ========================================================
    # PATIENT PRECAUTIONS ONLY
    # ========================================================

    if user_type == "Patient":

        st.subheader(
            "🛡️ General Precautions"
        )

        st.caption(
            "General precautions only. Follow advice from a qualified healthcare professional."
        )

        for item in precaution_items:

            st.write(
                f"• {item}"
            )


# ============================================================
# USER TYPE / NAVIGATION
# ============================================================

if page == "Prediction":
    st.markdown('<div class="nir-prediction-title">Risk Screening</div>', unsafe_allow_html=True)

    if st.session_state.get("logged_in") and st.session_state.get("logged_user"):
        logged_user = st.session_state["logged_user"]
        user_type = logged_user["user_type"]
        st.info(f"👤 Logged in as **{logged_user['name']}** · {logged_user['user_id']}")

        if st.button("🚪 Logout", key="logout_button"):
            st.session_state["logged_in"] = False
            st.session_state["logged_user"] = None
            st.session_state["page"] = "Home"
            st.rerun()
    else:
        user_type = st.radio(
            "Select User Type",
            ["Patient", "Doctor"],
            horizontal=True,
            key="user_type_selector"
        )
else:
    user_type = "Patient"


# ============================================================
# ABOUT SYSTEM
# ============================================================

if page == "About System":

    st.header(
        "ℹ️ About NIRAMAYA"
    )

    st.markdown(
        """
        ### What is NIRAMAYA?

        NIRAMAYA is a multi-disease screening platform
        that uses independent machine-learning models for
        different medical conditions.

        ### Available Models

        - Heart Disease
        - Parkinson's Disease
        - Diabetes
        - Chronic Kidney Disease
        - Heart Failure
        - Breast Cancer

        ### Machine Learning Workflow

        Dataset
        → Data Cleaning
        → EDA
        → Feature Selection
        → Train/Test Split
        → Standardization
        → Model Training
        → Hyperparameter Tuning
        → Evaluation
        → Model Saving
        → Streamlit Deployment

        ### Important

        The system is designed for screening support and
        educational/project purposes.

        Predictions should not be treated as a medical diagnosis.
        """
    )

    st.stop()


# ============================================================
# OTHER TOP-NAV PAGES
# ============================================================

if page == "How It Works":
    st.markdown('<div class="nir-section-title">How NIRAMAYA Works</div>', unsafe_allow_html=True)
    st.info("Dataset → Data Cleaning → Feature Selection → Standardization → Model Training → Evaluation → Screening Prediction → AI Guidance")
    st.markdown("""### 🤖 AI + Machine Learning
NIRAMAYA uses independent disease-specific models and optional AI medical-report OCR to assist with screening workflows.""")
    st.stop()

if page == "Health Insights":
    st.markdown('<div class="nir-section-title">Health Insights</div>', unsafe_allow_html=True)
    st.info("Use NIRAMAYA as a screening-support tool. Maintain regular checkups, healthy nutrition, physical activity, and professional medical follow-up.")
    st.stop()

if page == "Contact Us":
    st.markdown('<div class="nir-section-title">Contact NIRAMAYA</div>', unsafe_allow_html=True)
    st.markdown("For project demonstrations, technical feedback, or healthcare workflow suggestions, use the project contact channel configured by your team.")
    st.stop()

# ============================================================
# PATIENT PAGE
# ============================================================

if page == "Prediction" and user_type == "Patient":

    st.markdown(
        '<div class="nir-section-title-highlight">👤 Patient Portal</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # PATIENT LOGIN / REGISTER
    # --------------------------------------------------------
    if not st.session_state.get("logged_in"):

        login_tab, register_tab = st.tabs(["🔐 Patient Login", "📝 Patient Register"])

        with login_tab:
            st.subheader("Open Your NIRAMAYA Patient ID")
            patient_login_id = st.text_input(
                "Patient ID",
                placeholder="Example: NIR-P-XXXXXXXX",
                key="patient_portal_login_id"
            )
            patient_login_password = st.text_input(
                "Password",
                type="password",
                key="patient_portal_login_password"
            )

            if st.button("🔓 Open My Patient Account", type="primary", key="patient_portal_login_button"):
                user = login_user(patient_login_id, patient_login_password, "Patient")
                if user:
                    st.session_state["logged_in"] = True
                    st.session_state["logged_user"] = user
                    st.success(f"Welcome, {user['name']}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid Patient ID or password.")

        with register_tab:
            st.subheader("Create Your Patient Account")

            reg_name = st.text_input("Full Name", key="patient_portal_register_name")
            reg_gender = st.selectbox(
                "Gender", ["Male", "Female", "Other"],
                key="patient_portal_register_gender"
            )
            reg_age = st.number_input(
                "Age", min_value=1, max_value=120, value=25,
                key="patient_portal_register_age"
            )
            reg_password = st.text_input(
                "Create Password", type="password",
                key="patient_portal_register_password"
            )
            reg_confirm = st.text_input(
                "Confirm Password", type="password",
                key="patient_portal_register_confirm"
            )

            st.info(
                "After registration, you can choose either **Upload Medical Report** or **Enter Data Manually** from your Patient Portal."
            )

            if st.button("🆔 Register Patient", type="primary", key="patient_portal_register_button"):
                if not reg_name.strip() or not reg_password:
                    st.error("Please enter your name and password.")
                elif reg_password != reg_confirm:
                    st.error("Passwords do not match.")
                elif len(reg_password) < 6:
                    st.error("Password must contain at least 6 characters.")
                else:
                    new_id, error = register_patient(
                        reg_name.strip(), reg_gender, int(reg_age), reg_password
                    )
                    if new_id:
                        user = login_user(new_id, reg_password, "Patient")
                        if user:
                            st.session_state["logged_in"] = True
                            st.session_state["logged_user"] = user
                            st.session_state["patient_registration_success"] = True
                            st.rerun()
                        else:
                            st.success("✅ Patient account created successfully!")
                            st.info(f"🆔 Your NIRAMAYA Patient ID: **{new_id}**")
                    else:
                        st.error(f"❌ {error}")

    else:
        # ----------------------------------------------------
        # LOGGED-IN PATIENT PROFILE
        # ----------------------------------------------------
        logged_user = st.session_state["logged_user"]
        patient_id = logged_user["user_id"]

        st.success(f"🆔 Your NIRAMAYA Patient ID: **{patient_id}**")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Patient", logged_user.get("name", "-"))
        with c2:
            st.metric("Gender", logged_user.get("gender", "-"))
        with c3:
            st.metric("Age", logged_user.get("age", "-"))

        if st.session_state.pop("patient_registration_success", False):
            st.success("🎉 Your NIRAMAYA Patient Account is ready. You can now upload a report or enter parameters manually.")

        # ----------------------------------------------------
        # PATIENT ↔ DOCTOR MESSAGES
        # ----------------------------------------------------
        st.divider()
        st.subheader("💬 Message Your Doctor")
        st.caption("Optional: write a question or concern for the doctor. You can view the doctor's reply here later.")
        patient_message = st.text_area(
            "Your message",
            placeholder="Example: I uploaded my blood report. Could you please review it?",
            height=110,
            key="patient_message_box"
        )
        if st.button("📨 Send Message to Doctor", type="primary", key="patient_send_message_button"):
            if patient_message.strip():
                try:
                    save_patient_message(patient_id, logged_user.get("name", "Patient"), patient_message)
                    st.success("✅ Your message has been sent to the doctor.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Could not send message: {e}")
            else:
                st.warning("Please write a message first.")

        patient_messages = get_patient_messages(patient_id)
        if patient_messages:
            st.markdown("### 📨 Messages & Doctor Responses")
            for msg in patient_messages:
                mid, _, _, message, doctor_id_msg, doctor_name_msg, reply, created_at, replied_at = msg
                with st.expander(f"💬 Message · {created_at}", expanded=(reply is None)):
                    st.markdown(f"**You:** {message}")
                    if reply:
                        st.markdown(
                            f'<div class="doctor-reply-card"><b>👨‍⚕️ Dr. {doctor_name_msg or "Doctor"}</b>'
                            f'<br><span>{reply}</span><br><small>Replied: {replied_at or ""}</small></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("⏳ Waiting for a doctor response.")

        # ----------------------------------------------------
        # PATIENT SCREENING
        # ----------------------------------------------------
        st.divider()
        st.subheader("🩺 AI Health Screening")
        st.caption(
            "Choose a disease, enter or upload the patient's medical parameters, "
            "then run the AI screening model."
        )

        patient_disease = st.selectbox(
            "🔹 Select a Disease to Check Your Risk",
            list(MODEL_PATHS.keys()),
            format_func=lambda x: f"{DISEASE_INFO[x]['icon']} {x}",
            key="patient_disease"
        )

        patient_input_method = st.radio(
            "📋 Medical Data Source",
            ["Enter Data Manually", "Upload Medical Report"],
            horizontal=True,
            key="patient_input_method"
        )

        # ----------------------------------------------------
        # MEDICAL REPORT UPLOAD + OCR
        # ----------------------------------------------------
        if patient_input_method == "Upload Medical Report":
            st.info(
                "📄 Upload a PDF or clear medical-report image. NIRMAYA will save it to your Patient ID and try to extract matching values."
            )

            patient_uploaded_report = st.file_uploader(
                "Upload Medical Report",
                type=["pdf", "png", "jpg", "jpeg", "webp"],
                help="PDF and image medical reports are supported.",
                key="patient_medical_report"
            )

            if patient_uploaded_report is not None:
                file_ext = Path(patient_uploaded_report.name).suffix.lower()
                if file_ext == ".pdf":
                    st.success(f"📄 PDF selected: **{patient_uploaded_report.name}**")
                else:
                    st.image(patient_uploaded_report, caption="Uploaded Medical Report", width="stretch")

                if st.button(
                    "💾 Save Report & Extract Parameters",
                    type="primary",
                    key="patient_extract_report_data"
                ):
                    try:
                        with st.spinner("🤖 NIRMAYA AI is processing your medical report..."):
                            saved_report = save_uploaded_patient_report(
                                patient_uploaded_report, patient_id, logged_user.get("name", "Patient")
                            )

                        st.session_state["patient_ocr_data"] = saved_report["extracted_text"]
                        st.session_state["patient_ocr_disease"] = patient_disease
                        st.session_state["patient_last_saved_report"] = saved_report["file_name"]

                        st.success(f"✅ Report saved to Patient ID **{patient_id}**: **{saved_report['file_name']}**")
                        if saved_report["extracted_text"]:
                            st.success("🤖 Report text extracted. Matching parameters will be pre-filled below.")
                        else:
                            st.warning("📄 Report was saved, but no readable text could be extracted automatically. Please enter parameters manually.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Could not save/process the medical report: {e}")

        # Show every previously uploaded report linked to this Patient ID.
        show_patient_reports(patient_id, "📂 My Uploaded Medical Reports")

        # ----------------------------------------------------
        # LOAD SELECTED PATIENT MODEL
        # ----------------------------------------------------
        try:
            patient_model, patient_scaler, patient_features = load_model(patient_disease)
        except Exception as e:
            st.error(f"❌ Could not load {patient_disease} model")
            st.exception(e)
            st.stop()

        # ----------------------------------------------------
        # APPLY OCR VALUES BEFORE CREATING WIDGETS
        # ----------------------------------------------------
        patient_ocr_data = st.session_state.get("patient_ocr_data")
        patient_ocr_disease = st.session_state.get("patient_ocr_disease")

        if patient_ocr_data is not None and patient_ocr_disease == patient_disease:
            st.info(
                "🤖 Matching report values are being used to pre-fill parameters. "
                "Verify every value; missing or unclear values must be entered manually."
            )
            apply_ocr_to_inputs(
                patient_disease,
                patient_features,
                patient_ocr_data,
                "patient"
            )

        # ----------------------------------------------------
        # PATIENT PARAMETERS
        # ----------------------------------------------------
        st.markdown("### 🧪 Medical Parameters")
        st.caption(
            "Enter or verify your clinical parameters. Do not guess missing medical values."
        )

        patient_values = create_disease_inputs(
            patient_disease,
            patient_features,
            "patient"
        )

        # ----------------------------------------------------
        # PATIENT PREDICTION
        # ----------------------------------------------------
        if st.button(
            f"🔍 Check {patient_disease} Risk",
            type="primary",
            key="patient_predict_button"
        ):
            try:
                prediction, probability = make_prediction(
                    patient_model,
                    patient_scaler,
                    patient_features,
                    patient_values
                )

                display_result(
                    patient_disease,
                    prediction,
                    probability,
                    "Patient",
                    patient_name=logged_user.get("name"),
                    patient_gender=logged_user.get("gender"),
                    patient_age=logged_user.get("age"),
                    input_data=dict(zip(patient_features, patient_values)),
                    patient_id=patient_id,
                    doctor_id=None
                )
            except Exception as e:
                st.error("❌ Prediction Error")
                st.exception(e)

        # ----------------------------------------------------
        # PATIENT HISTORY
        # ----------------------------------------------------
        st.divider()
        st.subheader("📋 My Previous Screening / Medical Records")

        history = get_patient_history_by_id(patient_id)

        if history:
            rows = []
            for record in history:
                rows.append({
                    "Date": record[9],
                    "Disease": record[6],
                    "Screening Result": record[7],
                    "Confidence": f"{record[8]:.2f}%" if record[8] is not None else "N/A",
                    "Doctor ID": record[2] or "Self",
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

            latest = history[0]
            st.markdown(
                f"### Current/latest screening indication: **{latest[6]} — {latest[7]}**"
            )
            st.caption(
                "This is a screening indication from the stored AI result, not a medical diagnosis."
            )
        else:
            st.info("No previous screening records are available yet.")

# DOCTOR PAGE
# ============================================================

if page == "Prediction" and user_type == "Doctor":

    st.markdown(
        '<div class="nir-section-title-highlight">👨‍⚕️ Doctor Portal</div>',
        unsafe_allow_html=True
    )

    # Doctor authentication is also kept inside the Doctor side.
    if not st.session_state.get("logged_in"):
        login_tab, register_tab = st.tabs(["🔐 Doctor Login", "📝 Doctor Register"])

        with login_tab:
            doctor_login_id = st.text_input(
                "Doctor ID", placeholder="Example: NIR-D-XXXXXXXX",
                key="doctor_portal_login_id"
            )
            doctor_login_password = st.text_input(
                "Password", type="password",
                key="doctor_portal_login_password"
            )

            if st.button("🔓 Login as Doctor", type="primary", key="doctor_portal_login_button"):
                user = login_user(doctor_login_id, doctor_login_password, "Doctor")
                if user:
                    st.session_state["logged_in"] = True
                    st.session_state["logged_user"] = user
                    st.success(f"Welcome, {user['name']}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid Doctor ID or password.")

        with register_tab:
            doctor_reg_name = st.text_input("Doctor Name", key="doctor_portal_register_name")
            doctor_reg_gender = st.selectbox(
                "Gender", ["Male", "Female", "Other"],
                key="doctor_portal_register_gender"
            )
            doctor_reg_domain = st.selectbox(
                "Medical Domain",
                ["Cardiology", "Neurology", "Diabetology", "Nephrology", "Oncology", "General Medicine"],
                key="doctor_portal_register_domain"
            )
            doctor_reg_password = st.text_input(
                "Create Password", type="password",
                key="doctor_portal_register_password"
            )
            doctor_reg_confirm = st.text_input(
                "Confirm Password", type="password",
                key="doctor_portal_register_confirm"
            )

            if st.button("🆔 Register Doctor", type="primary", key="doctor_portal_register_button"):
                if not doctor_reg_name.strip() or not doctor_reg_password:
                    st.error("Please enter doctor name and password.")
                elif doctor_reg_password != doctor_reg_confirm:
                    st.error("Passwords do not match.")
                elif len(doctor_reg_password) < 6:
                    st.error("Password must contain at least 6 characters.")
                else:
                    new_id, error = register_doctor(
                        doctor_reg_name.strip(), doctor_reg_gender, doctor_reg_domain, doctor_reg_password
                    )
                    if new_id:
                        st.success("✅ Doctor account created successfully!")
                        st.info(f"🆔 Your NIRAMAYA Doctor ID: **{new_id}**")
                    else:
                        st.error(f"❌ {error}")

    else:
        logged_doctor = st.session_state["logged_user"]
        doctor_id = logged_doctor["user_id"]

        st.success(
            f"🩺 Dr. {logged_doctor.get('name', '')} · "
            f"{logged_doctor.get('domain', 'Medical Practice')} · "
            f"Doctor ID: **{doctor_id}**"
        )

        # --------------------------------------------------------
        # PATIENT SEARCH
        # --------------------------------------------------------
        st.subheader("🔎 Search Patient")
        st.caption("Enter a Patient ID or patient name to view previous screening/medical records.")

        patient_search = st.text_input(
            "Patient ID or Patient Name",
            placeholder="NIR-P-XXXXXXXX or patient name",
            key="doctor_patient_search"
        )

        selected_patient = None

        if st.button("🔍 Search Patient", type="primary", key="doctor_search_patient_button"):
            query = patient_search.strip()
            if not query:
                st.warning("Please enter a Patient ID or patient name.")
            else:
                if query.upper().startswith("NIR-P-"):
                    patient_id_query = query.upper()
                    search_history = get_patient_history_by_id(patient_id_query)
                    registered_patient = get_patient_by_id(patient_id_query)
                else:
                    search_history = get_patient_history(query)
                    registered_patient = None
                    if search_history:
                        registered_patient = get_patient_by_id(search_history[0][1])
                    else:
                        registered_patient = get_patient_by_name(query)

                if search_history or registered_patient:
                    selected_patient = search_history[0] if search_history else None
                    selected_id = search_history[0][1] if search_history else registered_patient[0]
                    st.session_state["doctor_selected_patient_history"] = search_history
                    st.session_state["doctor_selected_patient_id"] = selected_id
                    st.session_state["doctor_registered_patient"] = registered_patient
                else:
                    st.session_state["doctor_selected_patient_history"] = []
                    st.session_state["doctor_selected_patient_id"] = None
                    st.session_state["doctor_registered_patient"] = None
                    st.warning("No registered patient or screening record found.")

        search_history = st.session_state.get("doctor_selected_patient_history", [])
        selected_patient_id = st.session_state.get("doctor_selected_patient_id")
        registered_patient = st.session_state.get("doctor_registered_patient")

        if selected_patient_id:
            st.success(f"✅ Patient found: **{selected_patient_id}**")

            if search_history:
                latest = search_history[0]
                patient_name_display = latest[3]
                patient_gender_display = latest[4]
                patient_age_display = latest[5]
                latest_disease_display = latest[6]
            elif registered_patient:
                patient_name_display = registered_patient[2]
                patient_gender_display = registered_patient[3] or "-"
                patient_age_display = registered_patient[4] or "-"
                latest_disease_display = "No screening yet"
            else:
                patient_name_display = "-"
                patient_gender_display = "-"
                patient_age_display = "-"
                latest_disease_display = "No screening yet"

            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric("Patient Name", patient_name_display)
            with p2:
                st.metric("Gender", patient_gender_display)
            with p3:
                st.metric("Age", patient_age_display)
            with p4:
                st.metric("Latest Disease", latest_disease_display)

            if search_history:
                st.subheader("📋 Previous Medical / Screening Records")
                rows = []
                for record in search_history:
                    rows.append({
                        "Date": record[9],
                        "Disease": record[6],
                        "Result": record[7],
                        "Confidence": f"{record[8]:.2f}%" if record[8] is not None else "N/A",
                        "Doctor ID": record[2] or "Self",
                        "Medical Data": record[10],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                latest = search_history[0]
                st.info(
                    f"📌 Latest stored screening indication: **{latest[6]} — {latest[7]}**. "
                    "This does not establish a clinical diagnosis."
                )
            else:
                st.info("ℹ️ This patient is registered, but has no screening result saved yet.")

            # Reports uploaded by the patient during or after registration.
            show_patient_reports(selected_patient_id, "📄 Patient Medical Reports")

            # ----------------------------------------------------
            # PATIENT ↔ DOCTOR MESSAGES
            # ----------------------------------------------------
            st.subheader("💬 Patient Messages")
            patient_messages_for_doctor = get_patient_messages(selected_patient_id)
            if not patient_messages_for_doctor:
                st.info("No messages from this patient yet.")
            else:
                for msg in patient_messages_for_doctor:
                    mid, _, msg_patient_name, message, msg_doctor_id, msg_doctor_name, reply, created_at, replied_at = msg
                    with st.container():
                        st.markdown(
                            f'<div class="patient-message-card"><b>👤 {msg_patient_name}</b>'
                            f'<small>{created_at}</small><p>{message}</p></div>',
                            unsafe_allow_html=True
                        )
                        if reply:
                            st.success(f"✅ Replied by Dr. {msg_doctor_name or doctor_id} on {replied_at or 'date unavailable'}")
                            st.markdown(f"**Doctor response:** {reply}")
                        else:
                            reply_text = st.text_area(
                                "Write your response",
                                placeholder="Type a response for the patient...",
                                height=90,
                                key=f"doctor_reply_{mid}"
                            )
                            if st.button("📤 Send Reply", type="primary", key=f"doctor_send_reply_{mid}"):
                                if reply_text.strip():
                                    try:
                                        reply_to_patient_message(mid, doctor_id, logged_doctor.get("name", "Doctor"), reply_text)
                                        st.success("✅ Response sent to the patient.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Could not send response: {e}")
                                else:
                                    st.warning("Please write a response first.")
                        st.divider()

        st.divider()

        # --------------------------------------------------------
st.divider()

st.caption(
    "🏥 NIRAMAYA • Multi-Disease AI Screening • "
    "For screening support only, not medical diagnosis."
)
# ==========================================
# ==========================================
st.divider()

st.caption(
    "🏥 NIRAMAYA • Multi-Disease AI Screening • "
    "For screening support only, not medical diagnosis."
)