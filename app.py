import streamlit as st

st.set_page_config(
    page_title="SME Advisor Nigeria",
    page_icon="💼",
    layout="centered"
)

import joblib
import json
import random
import os
import re
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

CONFIDENCE_THRESHOLD = 0.018

def train_model():
    with open("intents.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(pattern.lower())
            labels.append(intent["tag"])

    for p in ["what do you mean", "i don't get it", "huh", "random stuff",
              "tell me something", "blah blah", "xyzzy", "asdfgh",
              "not sure what to ask", "something else", "other"]:
        texts.append(p)
        labels.append("fallback")

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(texts)
    mdl = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    mdl.fit(X, labels)

    joblib.dump(mdl, "model.pkl")
    joblib.dump(vec, "vectorizer.pkl")
    return mdl, vec


def is_model_stale():
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        return True
    try:
        mdl = joblib.load("model.pkl")
        vec = joblib.load("vectorizer.pkl")
        vec.transform(["test"])
        mdl.predict_proba(vec.transform(["test"]))
        return False
    except Exception:
        return True


@st.cache_resource
def load_resources():
    if is_model_stale():
        st.info("⚙️ Setting up advisor model — just a moment...")
        mdl, vec = train_model()
    else:
        mdl = joblib.load("model.pkl")
        vec = joblib.load("vectorizer.pkl")

    with open("intents.json", "r", encoding="utf-8") as f:
        intents = json.load(f)

    return mdl, vec, intents


try:
    model, vectorizer, data = load_resources()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

FOLLOW_UPS = {
    "registration": [
        "What documents do I need to register a business in Nigeria?",
        "How long does CAC registration take?",
        "What's the difference between a sole proprietor and an LLC?",
    ],
    "funding": [
        "What government grants are available for SMEs in Nigeria?",
        "How do I apply for a Bank of Industry loan?",
        "What do investors look for in a Nigerian startup?",
    ],
    "marketing": [
        "How do I use WhatsApp Business for marketing?",
        "What's the best social media platform for Nigerian SMEs?",
        "How do I build a brand on a tight budget?",
    ],
    "sales": [
        "How do I increase repeat customers?",
        "What pricing strategy works best for small businesses?",
        "How do I sell more during slow seasons?",
    ],
    "pricing": [
        "How do I price my product for the Nigerian market?",
        "Should I charge in Naira or USD for exports?",
        "How do I handle rising costs without losing customers?",
    ],
    "tax": [
        "How do I register for VAT in Nigeria?",
        "What are the FIRS filing deadlines for SMEs?",
        "Can I get a tax exemption as a small business?",
    ],
    "customer_service": [
        "How do I handle customer complaints professionally?",
        "How can I use CRM tools as a small business?",
        "What makes customers loyal to Nigerian brands?",
    ],
    "cash_flow": [
        "How do I manage cash flow during slow months?",
        "What's the best way to invoice clients promptly?",
        "How do I avoid running out of working capital?",
    ],
    "digital": [
        "How do I set up an online store in Nigeria?",
        "What digital payment options should my business accept?",
        "How do I use social media automation tools?",
    ],
    "fallback": [
        "How do I register my business in Nigeria?",
        "What funding options are available for SMEs?",
        "How can I market my business online?",
    ],
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700;800&display=swap');

:root {
    --bg:             #080E1A;
    --surface:        rgba(15, 23, 42, 0.85);
    --surface-solid:  #0F1929;
    --glass:          rgba(255,255,255,0.04);
    --glass-border:   rgba(255,255,255,0.08);
    --border:         #1E293B;
    --border-soft:    #162032;
    --accent:         #16A34A;
    --accent-2:       #059669;
    --accent-soft:    rgba(22,163,74,0.15);
    --accent-glow:    rgba(22,163,74,0.20);
    --accent-glow-lg: rgba(22,163,74,0.08);
    --text-primary:   #F1F5F9;
    --text-secondary: #94A3B8;
    --text-muted:     #64748B;
    --text-faint:     #334155;
    --gold:           #F59E0B;
    --radius:         14px;
    --radius-sm:      9px;
    --radius-lg:      20px;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-primary) !important;
}

.stApp::before {
    content: '';
    position: fixed;
    top: -200px;
    left: 50%;
    transform: translateX(-50%);
    width: 700px;
    height: 500px;
    background: radial-gradient(ellipse, rgba(22,163,74,0.07) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem !important;
    max-width: 800px !important;
    position: relative;
    z-index: 1;
}

.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,
.stMarkdown h4,.stMarkdown h5,.stMarkdown h6,
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: #F1F5F9 !important;
}
.stMarkdown strong,[data-testid="stChatMessage"] strong {
    color: #FFFFFF !important;
    font-weight: 600;
}

[data-testid="stChatMessage"] {
    background: var(--glass) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius) !important;
    padding: 16px 18px !important;
    margin-bottom: 12px !important;
    transition: box-shadow 0.25s ease, border-color 0.25s ease !important;
}

[data-testid="stChatMessage"]:hover {
    border-color: rgba(22,163,74,0.30) !important;
    box-shadow: 0 0 24px rgba(22,163,74,0.08), 0 4px 20px rgba(0,0,0,0.25) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(22,163,74,0.12) 0%, rgba(5,150,105,0.08) 100%) !important;
    border-color: rgba(22,163,74,0.22) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]):hover {
    box-shadow: 0 0 28px rgba(22,163,74,0.14), 0 4px 20px rgba(0,0,0,0.2) !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #14532D 0%, #16A34A 100%) !important;
    border: 1px solid rgba(22,163,74,0.4) !important;
    box-shadow: 0 0 12px rgba(22,163,74,0.25) !important;
}
[data-testid="chatAvatarIcon-user"] {
    background: var(--border) !important;
}

.stMarkdown p  { font-size: 14px !important; line-height: 1.75 !important; margin-bottom: 8px !important; }
.stMarkdown li { font-size: 14px !important; line-height: 1.7  !important; }
.stMarkdown ul,.stMarkdown ol { padding-left: 18px !important; margin-bottom: 8px !important; }
.stMarkdown code {
    background: rgba(22,163,74,0.12) !important;
    color: #4ADE80 !important;
    padding: 2px 7px !important;
    border-radius: 4px !important;
    font-size: 13px !important;
    border: 1px solid rgba(22,163,74,0.15) !important;
}

.sme-header {
    background: linear-gradient(135deg,
        rgba(13,36,25,0.95) 0%,
        rgba(8,14,26,0.90) 60%,
        rgba(5,150,105,0.08) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-left: 3px solid transparent;
    border-image: linear-gradient(180deg, #16A34A, #059669) 1;
    border-radius: var(--radius);
    padding: 26px 30px 22px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}
.sme-header::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(22,163,74,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.sme-header .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(22,163,74,0.15);
    border: 1px solid rgba(22,163,74,0.25);
    color: #4ADE80;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 12px;
}
.sme-header .logo-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}
.sme-header .logo-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #16A34A, #059669);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 0 20px rgba(22,163,74,0.30);
    flex-shrink: 0;
}
.sme-header h1 {
    font-family: 'Sora', sans-serif;
    font-size: 27px;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0;
    line-height: 1.2;
    background: linear-gradient(135deg, #FFFFFF 60%, #4ADE80 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sme-header p {
    color: var(--text-secondary) !important;
    font-size: 13.5px;
    margin: 0;
    line-height: 1.55;
}
.sme-header .powered {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    margin-top: 12px;
    font-size: 10.5px;
    color: var(--text-muted) !important;
    letter-spacing: 0.04em;
}
.sme-header .powered span { color: #4ADE80 !important; font-weight: 600; }

.insight-box {
    background: linear-gradient(135deg,
        rgba(22,163,74,0.08) 0%,
        rgba(5,150,105,0.05) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(22,163,74,0.18);
    border-radius: var(--radius);
    padding: 14px 18px;
    margin-bottom: 18px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.insight-box .insight-icon {
    font-size: 20px;
    flex-shrink: 0;
    margin-top: 1px;
}
.insight-box .insight-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #4ADE80 !important;
    margin-bottom: 3px;
}
.insight-box .insight-body {
    font-size: 13px;
    color: var(--text-secondary) !important;
    line-height: 1.55;
}

@keyframes blink {
    0%,80%,100% { opacity: 0.2; transform: scale(0.85); }
    40%         { opacity: 1;   transform: scale(1.1);  }
}
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 14px 18px;
    background: var(--glass);
    backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    margin-bottom: 12px;
    width: fit-content;
}
.typing-dot {
    width: 7px; height: 7px;
    background: #4ADE80;
    border-radius: 50%;
    animation: blink 1.2s infinite ease-in-out;
    box-shadow: 0 0 6px rgba(74,222,128,0.5);
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
.typing-label {
    font-size: 11.5px;
    color: var(--text-muted);
    margin-left: 4px;
    letter-spacing: 0.02em;
}

.suggestions-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 14px 0 4px;
}
.suggestion-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 6px 13px;
    background: rgba(22,163,74,0.07);
    border: 1px solid rgba(22,163,74,0.20);
    border-radius: 20px;
    font-size: 12px;
    color: #86EFAC !important;
    cursor: pointer;
    transition: all 0.18s;
    font-weight: 500;
}
.suggestion-chip:hover {
    background: rgba(22,163,74,0.18);
    border-color: rgba(22,163,74,0.45);
    box-shadow: 0 0 14px rgba(22,163,74,0.15);
    color: #4ADE80 !important;
}

.chips-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 20px;
}
.q-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 6px 14px;
    background: var(--glass);
    backdrop-filter: blur(8px);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    font-size: 12px;
    color: var(--text-secondary);
    transition: all 0.18s;
}
.q-chip:hover {
    border-color: rgba(22,163,74,0.35);
    color: #4ADE80;
    background: rgba(22,163,74,0.08);
    box-shadow: 0 0 12px rgba(22,163,74,0.12);
}

.conf-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 10.5px;
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 20px;
    margin-top: 10px;
    letter-spacing: 0.02em;
}
.conf-chip.high { background: rgba(22,163,74,0.14); color: #4ADE80; border: 1px solid rgba(22,163,74,0.20); }
.conf-chip.mid  { background: rgba(245,158,11,0.10); color: #F59E0B; border: 1px solid rgba(245,158,11,0.18); }
.conf-chip.low  { background: rgba(239,68,68,0.08);  color: #F87171; border: 1px solid rgba(239,68,68,0.15); }

.msg-ts {
    font-size: 10px;
    color: var(--text-faint);
    margin-top: 8px;
    display: block;
}

[data-testid="stSidebar"] {
    background: rgba(10,16,30,0.92) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid var(--glass-border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }

.sidebar-brand {
    font-family: 'Sora', sans-serif;
    font-size: 15px;
    font-weight: 800;
    background: linear-gradient(135deg, #FFFFFF 50%, #4ADE80 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 2px;
}
.sidebar-tagline { font-size: 11px; color: var(--text-muted) !important; margin-bottom: 2px; }
.sidebar-section-title {
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-faint) !important;
    margin: 16px 0 8px;
}
.topic-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    border-radius: 7px;
    font-size: 12.5px;
    color: var(--text-secondary) !important;
    margin-bottom: 3px;
    transition: all 0.15s;
}
.topic-pill:hover {
    background: rgba(22,163,74,0.07);
    color: #86EFAC !important;
}
.topic-pill .dot {
    width: 5px; height: 5px;
    background: var(--accent);
    border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0 0 6px rgba(22,163,74,0.4);
}

.stat-card {
    background: rgba(22,163,74,0.05);
    border: 1px solid rgba(22,163,74,0.12);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    margin-top: 6px;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    color: var(--text-muted) !important;
    padding: 3px 0;
}
.stat-val { color: #4ADE80 !important; font-weight: 700; }

.memory-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.15);
    border-radius: var(--radius-sm);
    font-size: 12px;
    color: #FCD34D !important;
    margin-bottom: 6px;
}

/* ── CHAT INPUT — black text on white background ── */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    backdrop-filter: none !important;
    border: 1px solid rgba(22,163,74,0.30) !important;
    border-radius: var(--radius) !important;
    transition: all 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(22,163,74,0.65) !important;
    box-shadow: 0 0 0 3px rgba(22,163,74,0.10), 0 0 20px rgba(22,163,74,0.08) !important;
}
[data-testid="stChatInput"] textarea {
    color: #000000 !important;
    background: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    caret-color: #000000 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #888888 !important; }

.stButton > button {
    background: transparent !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--text-secondary) !important;
    font-size: 12.5px !important;
    font-family: 'Inter', sans-serif !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px 14px !important;
    width: 100% !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    border-color: rgba(22,163,74,0.4) !important;
    color: #4ADE80 !important;
    background: rgba(22,163,74,0.07) !important;
    box-shadow: 0 0 14px rgba(22,163,74,0.10) !important;
}

[data-testid="stSelectbox"] label { color: var(--text-secondary) !important; font-size: 12px !important; }
[data-testid="stSelectbox"] > div > div {
    background: var(--surface-solid) !important;
    border-color: var(--border) !important;
    color: #FFFFFF !important;
    font-size: 13px !important;
}

[data-testid="stExpander"] {
    background: var(--glass) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stExpander"] summary { color: var(--text-secondary) !important; font-size: 13px !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

hr { border-color: var(--border) !important; margin: 14px 0 !important; }

.grad-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(22,163,74,0.35), transparent);
    margin: 16px 0;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sme-header">
    <div class="badge">🇳🇬 Nigeria &nbsp;·&nbsp; SME Intelligence</div>
    <div class="logo-row">
        <div class="logo-icon">💼</div>
        <h1>SME Business Advisor</h1>
    </div>
    <p>Expert guidance on registration, funding, marketing, compliance, and growth — built for Nigerian entrepreneurs.</p>
    <div class="powered">Powered by &nbsp;<span> Timothy.Ogunleye </span>&nbsp;·&nbsp; Local Business Intelligence Engine</div>
</div>
""", unsafe_allow_html=True)

DAILY_INSIGHTS = [
    ("📊", "Nigeria has over 41 million SMEs, contributing ~48% of GDP. Formalising your business opens access to grants, loans, and government contracts."),
    ("💡", "WhatsApp Business has over 50 million active users in Nigeria — it's the highest-ROI marketing channel for SMEs right now."),
    ("📈", "NIRSAL Microfinance Bank offers loans from ₦50,000 to ₦3 million for micro-businesses with no collateral required."),
    ("🔒", "Registered businesses pay lower effective tax rates due to legitimate deductions — compliance is a growth strategy, not just an obligation."),
    ("🤝", "80% of Nigerian SME sales still come from referrals. A structured loyalty or referral programme can 3× your growth rate."),
]

day_index = datetime.now().day % len(DAILY_INSIGHTS)
insight_icon, insight_text = DAILY_INSIGHTS[day_index]

st.markdown(f"""
<div class="insight-box">
    <div class="insight-icon">{insight_icon}</div>
    <div>
        <div class="insight-title">Today's Business Intelligence</div>
        <div class="insight-body">{insight_text}</div>
    </div>
</div>
""", unsafe_allow_html=True)

QUICK_QUESTIONS = [
    ("🏛️", "How do I register my business?"),
    ("💰", "How can I get funding or a loan?"),
    ("📋", "What taxes does my business owe?"),
    ("📣", "Tips for marketing my small business"),
    ("💳", "How do I manage cash flow?"),
    ("📈", "How can I grow my sales?"),
]

chips_html = '<div class="chips-row">'
for icon, q in QUICK_QUESTIONS:
    chips_html += f'<span class="q-chip">{icon} {q}</span>'
chips_html += '</div>'
st.markdown(chips_html, unsafe_allow_html=True)

TOPICS = [
    ("🏛️", "Business Registration"),
    ("💰", "Funding & Loans"),
    ("📣", "Marketing"),
    ("📈", "Sales Growth"),
    ("🏷️", "Pricing Strategy"),
    ("📋", "Tax & Compliance"),
    ("🤝", "Customer Service"),
    ("💳", "Cash Flow"),
    ("💻", "Digital Transformation"),
]

with st.sidebar:
    st.markdown('<div class="sidebar-brand">SME Advisor 🇳🇬</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Powered by local business intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sidebar-section-title">Topics I can help with</div>', unsafe_allow_html=True)
    for icon, label in TOPICS:
        st.markdown(f'<div class="topic-pill"><span class="dot"></span><span>{icon} {label}</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    msgs       = st.session_state.get("messages", [])
    user_msgs  = [m for m in msgs if m["role"] == "user"]
    avg_conf   = st.session_state.get("avg_confidence", 0.0)
    topics_hit = st.session_state.get("topics_hit", set())

    st.markdown('<div class="sidebar-section-title">Session Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-row"><span>Messages sent</span><span class="stat-val">{len(user_msgs)}</span></div>
        <div class="stat-row"><span>Topics covered</span><span class="stat-val">{len(topics_hit)}</span></div>
        <div class="stat-row"><span>Avg confidence</span><span class="stat-val">{avg_conf:.0%}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("📥 Export conversation"):
        if len(msgs) > 1:
            lines = []
            for m in msgs:
                role = "You" if m["role"] == "user" else "Advisor"
                content = re.sub(r"<[^>]+>", "", m["content"])
                lines.append(f"[{role}]\n{content}\n")
            export_text = "\n---\n".join(lines)
            st.download_button(
                label="💾 Download as .txt",
                data=export_text,
                file_name=f"sme_advisor_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

    if st.button("🗑️ Clear conversation"):
        for key in ["messages","avg_confidence","topics_hit","confidence_scores","memory_context"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.markdown("""
    <div style="margin-top:20px;font-size:11px;color:#334155;line-height:1.6;">
        Advice is for general guidance only.<br>
        Consult a professional for legal or<br>financial decisions.
    </div>
    """, unsafe_allow_html=True)


def predict_intent(text):
    X = vectorizer.transform([text.lower().strip()])
    probs = model.predict_proba(X)[0]
    confidence = float(max(probs))
    tag = str(model.classes_[probs.argmax()])
    if confidence < CONFIDENCE_THRESHOLD:
        return "fallback", confidence
    return tag, confidence


def get_response(tag):
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    for intent in data["intents"]:
        if intent["tag"] == "fallback":
            return random.choice(intent["responses"])
    return (
        "I want to make sure I give you the right advice, but I didn't quite catch what you're asking.\n\n"
        "Here are the areas I know best — try asking something like:\n\n"
        "- **\"How do I register my business in Nigeria?\"**\n"
        "- **\"What funding options are available for SMEs?\"**\n"
        "- **\"How should I price my products?\"**\n"
        "- **\"What taxes does my business owe?\"**\n\n"
        "You can also tap one of the quick-question chips above ☝️"
    )


def confidence_label(score: float):
    if score >= 0.6:
        return "high", f"High confidence · {score:.0%}"
    elif score >= 0.3:
        return "mid",  f"Moderate confidence · {score:.0%}"
    else:
        return "low",  f"Low confidence · {score:.0%}"


TOPIC_MAP = {
    "registration":     "Business Registration",
    "funding":          "Funding & Loans",
    "marketing":        "Marketing",
    "sales":            "Sales Growth",
    "pricing":          "Pricing Strategy",
    "tax":              "Tax & Compliance",
    "customer_service": "Customer Service",
    "cash_flow":        "Cash Flow",
    "digital":          "Digital Transformation",
}


def format_response(response: str, tag: str, confidence: float) -> str:
    conf_class, conf_text = confidence_label(confidence)
    ts    = datetime.now().strftime("%I:%M %p")
    topic = TOPIC_MAP.get(tag)
    topic_html = (
        f'<span style="font-size:10.5px;color:#4ADE80;margin-left:8px;'
        f'background:rgba(22,163,74,0.10);padding:2px 8px;border-radius:10px;">'
        f'↗ {topic}</span>'
    ) if topic else ""

    return (
        f"{response}\n\n"
        f'<div class="grad-divider"></div>'
        f'<span class="conf-chip {conf_class}">'
        f'{"✅" if conf_class=="high" else "⚠️" if conf_class=="mid" else "❓"} {conf_text}'
        f'</span>{topic_html}'
        f'<span class="msg-ts">{ts}</span>'
    )


def get_suggestions(tag: str) -> list:
    pool = FOLLOW_UPS.get(tag, FOLLOW_UPS["fallback"])
    return random.sample(pool, min(2, len(pool)))


def build_memory_context() -> str:
    topics = st.session_state.get("topics_hit", set())
    if not topics:
        return ""
    readable = [TOPIC_MAP.get(t, t) for t in topics]
    return f"The user has already asked about: {', '.join(readable)}."


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome! I'm your **SME Business Advisor**, built specifically for Nigerian entrepreneurs.\n\n"
                "I can help you with **business registration**, **funding and loans**, **marketing**, "
                "**sales growth**, **pricing**, **tax compliance**, **cash flow**, and more.\n\n"
                "What challenge is your business facing today? 👇"
            )
        }
    ]

for key, default in [
    ("confidence_scores", []),
    ("topics_hit", set()),
    ("avg_confidence", 0.0),
    ("last_tag", None),
    ("show_suggestions", False),
    ("pending_suggestions", []),
    ("memory_context", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.topics_hit:
    mem = build_memory_context()
    st.markdown(
        f'<div class="memory-banner">🧠 &nbsp;{mem} I\'ll use this to personalise my answers.</div>',
        unsafe_allow_html=True
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

if st.session_state.show_suggestions and st.session_state.pending_suggestions:
    sugg_html = '<div class="suggestions-row">💬 Follow up: &nbsp;'
    for s in st.session_state.pending_suggestions:
        sugg_html += f'<span class="suggestion-chip">→ {s}</span>'
    sugg_html += '</div>'
    st.markdown(sugg_html, unsafe_allow_html=True)

with st.expander("💡 Quick questions — tap to ask", expanded=False):
    cols = st.columns(2)
    for i, (icon, question) in enumerate(QUICK_QUESTIONS):
        with cols[i % 2]:
            if st.button(f"{icon} {question}", key=f"quick_{i}"):
                st.session_state._quick_q = question
                st.rerun()

if hasattr(st.session_state, "_quick_q") and st.session_state._quick_q:
    user_input = st.session_state._quick_q
    st.session_state._quick_q = None
else:
    user_input = st.chat_input("Ask about your business…")

if user_input:
    st.session_state.show_suggestions = False
    st.session_state.pending_suggestions = []

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
    <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <span class="typing-label">Advisor is thinking…</span>
    </div>
    """, unsafe_allow_html=True)

    time.sleep(0.8)
    typing_placeholder.empty()

    tag, confidence = predict_intent(user_input)
    raw_response    = get_response(tag)
    full_response   = format_response(raw_response, tag, confidence)

    st.session_state.confidence_scores.append(confidence)
    st.session_state.avg_confidence = (
        sum(st.session_state.confidence_scores) / len(st.session_state.confidence_scores)
    )
    if tag != "fallback":
        st.session_state.topics_hit.add(tag)

    st.session_state.last_tag = tag
    st.session_state.memory_context = build_memory_context()

    with st.chat_message("assistant"):
        msg_placeholder = st.empty()

        displayed = ""
        chunk_size = 8
        for i in range(0, len(raw_response), chunk_size):
            displayed += raw_response[i:i + chunk_size]
            msg_placeholder.markdown(displayed + "▌", unsafe_allow_html=True)
            time.sleep(0.012)

        msg_placeholder.markdown(full_response, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    suggestions = get_suggestions(tag)
    st.session_state.pending_suggestions = suggestions
    st.session_state.show_suggestions    = True

    st.rerun()

