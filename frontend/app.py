import streamlit as st
import requests
import json
import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="MindLens",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# GLOBAL CSS – DARK THEME
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --surface2:  #181c24;
    --border:    #1f2430;
    --accent:    #6ee7b7;          /* emerald */
    --accent2:   #818cf8;          /* indigo  */
    --danger:    #f87171;
    --warn:      #fbbf24;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

/* ── Base ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-mono);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1280px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Inputs ── */
input, textarea, select,
.stTextInput > div > div > input,
.stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}
input:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(110,231,183,.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--font-mono) !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: .5rem 1.4rem !important;
    transition: all .2s ease;
    letter-spacing: .04em;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
}

/* ── File uploader ── */
.stFileUploader > div {
    background: var(--surface2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: var(--font-head); font-size: 2rem; color: var(--accent) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border);
    gap: .5rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: var(--font-mono) !important;
    border-radius: 6px 6px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface2) !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Progress / slider ── */
.stProgress > div > div > div { background: var(--accent) !important; }

/* ── Alerts ── */
.stAlert { border-radius: 10px !important; font-family: var(--font-mono) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Custom card ── */
.ml-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.ml-card-accent { border-left: 3px solid var(--accent); }
.ml-card-danger { border-left: 3px solid var(--danger); }
.ml-card-warn   { border-left: 3px solid var(--warn);   }

/* ── Score badge ── */
.score-pill {
    display: inline-block;
    padding: .2rem .8rem;
    border-radius: 999px;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .06em;
    font-family: var(--font-mono);
}
.score-low  { background: rgba(110,231,183,.15); color: #6ee7b7; border:1px solid #6ee7b7; }
.score-mid  { background: rgba(251,191,36,.12);  color: #fbbf24; border:1px solid #fbbf24; }
.score-high { background: rgba(248,113,113,.12); color: #f87171; border:1px solid #f87171; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #111318 0%, #0f1620 60%, #0a1520 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(110,231,183,.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: var(--font-head);
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -.02em;
    background: linear-gradient(90deg, #6ee7b7, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 .4rem;
}
.hero-sub { color: var(--muted); font-size: .85rem; letter-spacing: .04em; }

/* ── Word-impact bar ── */
.word-bar-wrap { display: flex; align-items: center; gap: .6rem; margin: .3rem 0; }
.word-label { min-width: 120px; font-size: .78rem; color: var(--text); }
.word-bar-bg { flex: 1; background: var(--surface2); border-radius: 4px; height: 6px; overflow: hidden; }
.word-bar-fill-pos { height: 6px; background: var(--accent); border-radius: 4px; }
.word-bar-fill-neg { height: 6px; background: var(--danger); border-radius: 4px; }
.word-val { font-size: .72rem; color: var(--muted); min-width: 50px; text-align: right; }

/* ── History row ── */
.hist-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: .65rem 1rem;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: .5rem;
    font-size: .8rem;
}
.hist-ts { color: var(--muted); font-size: .7rem; }

/* ── Emotion tag ── */
.emotion-tag {
    font-size: .7rem;
    padding: .15rem .6rem;
    border-radius: 6px;
    background: var(--surface2);
    border: 1px solid var(--border);
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--text);
}

/* ── Section label ── */
.section-label {
    font-family: var(--font-head);
    font-size: .65rem;
    text-transform: uppercase;
    letter-spacing: .14em;
    color: var(--muted);
    margin-bottom: .6rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
for k, v in [
    ("token", None),
    ("user", None),
    ("last_fusion", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}

def risk_pill(score):
    if score is None:
        return ""
    if score < 35:
        cls, label = "score-low", "LOW"
    elif score < 65:
        cls, label = "score-mid", "MODERATE"
    else:
        cls, label = "score-high", "HIGH"
    return f'<span class="score-pill {cls}">{label}</span>'

def fmt_ts(ts_str):
    try:
        dt = datetime.fromisoformat(str(ts_str).replace("Z", ""))
        return dt.strftime("%d %b %Y  %H:%M")
    except Exception:
        return str(ts_str)

# ─────────────────────────────────────────
# API CALLS
# ─────────────────────────────────────────
def api_signup(name, email, password):
    try:
        r = requests.post(f"{API_BASE}/signup",
                          json={"name": name, "email": email, "password": password})
        return r.json(), r.status_code
    except requests.exceptions.RequestException:
        return {"detail": "Backend is offline."}, 503

def api_login(email, password):
    try:
        r = requests.post(f"{API_BASE}/login",
                          json={"email": email, "password": password})
        return r.json(), r.status_code
    except requests.exceptions.RequestException:
        return {"detail": "Backend is offline."}, 503

def api_me():
    try:
        r = requests.get(f"{API_BASE}/me", headers=auth_headers())
        return r.json()
    except requests.exceptions.RequestException:
        return {"name": "Offline User", "email": "offline@example.com"}

def api_predict_text(text):
    try:
        r = requests.post(f"{API_BASE}/predict/text", json={"text": text})
        return r.json()
    except requests.exceptions.RequestException:
        return {"detail": "Backend is offline."}

def api_explain_text(text):
    try:
        r = requests.post(f"{API_BASE}/explain/text", json={"text": text})
        return r.json()
    except requests.exceptions.RequestException:
        return {"detail": "Backend is offline."}

def api_predict_audio(file_bytes, filename):
    try:
        r = requests.post(f"{API_BASE}/predict/audio",
                          files={"file": (filename, file_bytes, "audio/wav")})
        return r.json()
    except requests.exceptions.RequestException:
        return {"detail": "Backend is offline."}

def api_predict_face(file_bytes, filename):
    try:
        r = requests.post(f"{API_BASE}/predict/face",
                          files={"file": (filename, file_bytes, "image/jpeg")})
        return r.json()
    except requests.exceptions.RequestException:
        return {"detail": "Backend is offline."}

def api_fusion(text=None, audio_bytes=None, audio_name=None,
               face_bytes=None, face_name=None):
    data, files = {}, {}
    if text:
        data["text"] = text
    if audio_bytes:
        files["audio"] = (audio_name, audio_bytes, "audio/wav")
    if face_bytes:
        files["face"] = (face_name, face_bytes, "image/jpeg")
    try:
        r = requests.post(f"{API_BASE}/predict/fusion",
                          data=data, files=files, headers=auth_headers())
        return r.json(), r.status_code
    except requests.exceptions.RequestException:
        return {"detail": "Backend is offline."}, 503

def api_report():
    try:
        r = requests.get(f"{API_BASE}/mental-health-report", headers=auth_headers())
        return r.json()
    except requests.exceptions.RequestException:
        return {"detail": "Backend is offline."}

def api_trend():
    try:
        r = requests.get(f"{API_BASE}/trend", headers=auth_headers())
        return r.json()
    except requests.exceptions.RequestException:
        return {"detail": "Backend is offline."}

# ─────────────────────────────────────────
# COMPONENTS
# ─────────────────────────────────────────
def hero_banner():
    st.markdown("""
    <div class="hero">
        <div class="hero-title">MindLens</div>
        <div class="hero-sub">Multimodal Mental Health Analysis · v5.0</div>
    </div>
    """, unsafe_allow_html=True)

def render_risk_gauge(score, label="Risk Score"):
    if score is None:
        st.info("Not enough data for prediction yet.")
        return
    color = "#6ee7b7" if score < 35 else ("#fbbf24" if score < 65 else "#f87171")
    pct = score if score <= 1 else score  # handle 0-1 or 0-100
    if pct <= 1:
        pct = round(pct * 100, 1)
    st.markdown(f"""
    <div class="ml-card" style="text-align:center;">
        <div class="section-label">{label}</div>
        <div style="font-family:var(--font-head);font-size:3.2rem;font-weight:800;
                    color:{color};line-height:1;">{pct}</div>
        <div style="color:var(--muted);font-size:.75rem;margin:.3rem 0 1rem;">/ 100</div>
    """, unsafe_allow_html=True)
    st.progress(int(min(pct, 100)) / 100)
    st.markdown("</div>", unsafe_allow_html=True)

def render_emotion_result(result, title):
    if not result:
        return
    emotion = result.get("emotion", "—")
    conf    = round(result.get("confidence", 0) * 100, 1)
    risk    = result.get("risk_score", 0)
    st.markdown(f"""
    <div class="ml-card ml-card-accent">
        <div class="section-label">{title}</div>
        <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
            <span class="emotion-tag">{emotion}</span>
            <span style="color:var(--muted);font-size:.78rem;">confidence <b style="color:var(--text);">{conf}%</b></span>
            <span style="color:var(--muted);font-size:.78rem;">risk <b style="color:var(--text);">{risk}</b></span>
            {risk_pill(risk)}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_word_bars(words):
    if not words:
        return
    max_abs = max(abs(w["impact"]) for w in words) or 1
    st.markdown('<div class="section-label">Top Influential Words</div>', unsafe_allow_html=True)
    for w in words:
        val = w["impact"]
        pct = abs(val) / max_abs * 100
        cls = "word-bar-fill-pos" if val >= 0 else "word-bar-fill-neg"
        st.markdown(f"""
        <div class="word-bar-wrap">
            <span class="word-label">{w['word']}</span>
            <div class="word-bar-bg"><div class="{cls}" style="width:{pct:.1f}%"></div></div>
            <span class="word-val">{val:+.4f}</span>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# AUTH PAGE
# ─────────────────────────────────────────
def page_auth():
    hero_banner()
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["🔑  Login", "✨  Sign Up"])

        with tab_login:
            st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
            email    = st.text_input("Email", key="li_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="li_pass")
            if st.button("Login", use_container_width=True):
                if email and password:
                    with st.spinner("Authenticating…"):
                        data, code = api_login(email, password)
                    if code == 200:
                        st.session_state.token = data["access_token"]
                        me = api_me()
                        st.session_state.user = me
                        st.success(f"Welcome back, {me['name']} 👋")
                        st.rerun()
                    else:
                        st.error(data.get("detail", "Login failed"))
                else:
                    st.warning("Please fill in all fields.")

        with tab_signup:
            st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
            name     = st.text_input("Full Name", key="su_name")
            email2   = st.text_input("Email", key="su_email", placeholder="you@example.com")
            pass2    = st.text_input("Password", type="password", key="su_pass")
            if st.button("Create Account", use_container_width=True):
                if name and email2 and pass2:
                    with st.spinner("Creating account…"):
                        data, code = api_signup(name, email2, pass2)
                    if code == 200:
                        st.success("Account created! Please log in.")
                    else:
                        st.error(data.get("detail", "Signup failed"))
                else:
                    st.warning("Please fill in all fields.")

# ─────────────────────────────────────────
# SIDEBAR (authenticated)
# ─────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:.8rem 0 1.4rem;">
            <div style="font-family:var(--font-head);font-size:1.3rem;font-weight:800;
                        background:linear-gradient(90deg,#6ee7b7,#818cf8);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                🧠 MindLens
            </div>
            <div style="color:var(--muted);font-size:.72rem;margin-top:.2rem;">v5.0 · Mental Health AI</div>
        </div>
        <hr>
        <div style="margin:.8rem 0 1.4rem;">
            <div style="font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;">Signed in as</div>
            <div style="font-size:.9rem;color:var(--text);margin-top:.2rem;">{st.session_state.user['name']}</div>
            <div style="font-size:.72rem;color:var(--muted);">{st.session_state.user['email']}</div>
        </div>
        <hr>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🔮  Fusion Analysis", "📊  Dashboard"],
            label_visibility="collapsed"
        )

        st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            st.session_state.token = None
            st.session_state.user  = None
            st.session_state.last_fusion = None
            st.rerun()

    return page

# ─────────────────────────────────────────
# PAGE: FUSION
# ─────────────────────────────────────────
def page_fusion():
    hero_banner()
    st.markdown('<div class="section-label">Provide one or more inputs for multimodal analysis</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**💬 Text Input**")
            text_in = st.text_area("Write your thoughts…", height=130, key="fusion_text",
                                   label_visibility="collapsed",
                                   placeholder="How are you feeling today?")
    with c2:
        with st.container(border=True):
            st.markdown("**🎙️ Voice / Audio**")
            audio_file = st.file_uploader("Upload .wav file", type=["wav"],
                                          key="fusion_audio", label_visibility="collapsed")
            st.markdown('<div style="text-align: center; margin: 8px 0; color: var(--muted); font-size: 0.8rem;">— OR —</div>', unsafe_allow_html=True)
            
            from streamlit_mic_recorder import mic_recorder
            recorded_audio = mic_recorder(
                start_prompt="Record Voice",
                stop_prompt="Stop Recording",
                just_once=False,
                use_container_width=True,
                format="wav",
                key="mic_recorder"
            )
            if recorded_audio:
                st.audio(recorded_audio["bytes"], format="audio/wav")
    with c3:
        with st.container(border=True):
            st.markdown("**📸 Face Image**")
            face_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"],
                                         key="fusion_face", label_visibility="collapsed")
            st.markdown('<div style="text-align: center; margin: 8px 0; color: var(--muted); font-size: 0.8rem;">— OR —</div>', unsafe_allow_html=True)
            
            enable_camera = st.toggle("📸 Enable Camera", value=False, key="enable_camera_toggle")
            camera_file = None
            if enable_camera:
                camera_file = st.camera_input("Take a photo", key="fusion_camera", label_visibility="collapsed")

    run = st.button("⚡  Run Fusion Analysis", use_container_width=True)

    if run:
        if not text_in and not audio_file and not recorded_audio and not face_file and not camera_file:
            st.warning("Provide at least one input.")
            return
        with st.spinner("Analysing…"):
            ab = None
            an = None
            if audio_file:
                ab = audio_file.read()
                an = audio_file.name
            elif recorded_audio:
                ab = recorded_audio["bytes"]
                an = "recorded_audio.wav"

            fb = None
            fn = None
            if face_file:
                fb = face_file.read()
                fn = face_file.name
            elif camera_file:
                fb = camera_file.read()
                fn = camera_file.name

            result, code = api_fusion(text_in or None, ab, an, fb, fn)
        if code != 200:
            st.error(result.get("detail", "Fusion failed."))
            return
        st.session_state.last_fusion = result
        st.rerun()

    res = st.session_state.last_fusion
    if not res:
        return

    st.markdown("---")
    st.markdown('<div class="section-label">Analysis Results</div>', unsafe_allow_html=True)

    final = res.get("final_result", {})
    future = res.get("future_risk")

    col_score, col_future = st.columns(2)
    with col_score:
        render_risk_gauge(final.get("risk_score"), "Overall Risk Score")
    with col_future:
        if future is not None:
            render_risk_gauge(round(future * 100, 1), "Predicted Future Risk")
        else:
            st.markdown("""
            <div class="ml-card" style="text-align:center;padding:2rem;">
                <div style="color:var(--muted);font-size:.8rem;">Future risk requires 7+ sessions.</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ml-card ml-card-accent" style="text-align:center;margin-top:.5rem;">
        <div class="section-label">Risk Level</div>
        <div style="font-family:var(--font-head);font-size:2rem;font-weight:800;
                    color:var(--accent);">{final.get('risk_level','—')}</div>
    </div>
    """, unsafe_allow_html=True)

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        render_emotion_result(res.get("text_result"),  "Text Modality")
    with ec2:
        render_emotion_result(res.get("audio_result"), "Audio Modality")
    with ec3:
        render_emotion_result(res.get("face_result"),  "Face Modality")

# ─────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────
def page_dashboard():
    st.markdown('<div class="hero-title" style="font-size:1.6rem;font-family:var(--font-head);font-weight:800;background:linear-gradient(90deg,#6ee7b7,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:1.2rem;">📊 Mental Health Dashboard</div>', unsafe_allow_html=True)

    col_rep, col_trend = st.columns(2)

    with col_rep:
        if st.button("Load Dashboard Data", use_container_width=True):
            with st.spinner("Fetching data…"):
                r = api_report()
            st.session_state["_report"] = r

    with col_trend:
        if st.button("Load Trend Forecast", use_container_width=True):
            with st.spinner("Fetching trend…"):
                t = api_trend()
            st.session_state["_trend"] = t

    # Trend card
    if "_trend" in st.session_state and st.session_state["_trend"]:
        t = st.session_state["_trend"]
        if "detail" in t:
            st.error(t["detail"])
        else:
            future = t.get("future_risk")
            pct = round(future * 100, 1) if future is not None else None
            st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
            render_risk_gauge(pct, "Predicted Future Risk (LSTM)")

    # Report & Graphs
    if "_report" in st.session_state and st.session_state["_report"]:
        r = st.session_state["_report"]
        if "detail" in r:
            st.error(r["detail"])
        else:
            future = r.get("future_risk")
            status = r.get("status", "—")
            history = r.get("history", [])
            records = r.get("records_found", 0)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Sessions Found", records)
            col_b.metric("Status", status)
            col_c.metric("Future Risk", f"{round(future*100,1):.1f}" if future else "N/A")

            if history:
                st.markdown("---")
                st.markdown('<div class="section-label">Historical Trends</div>', unsafe_allow_html=True)
                
                # Parse history for Pandas
                data = []
                for h in reversed(history):
                    dt = datetime.fromisoformat(str(h.get("timestamp")).replace("Z", ""))
                    data.append({
                        "Time": dt.strftime("%b %d %H:%M"),
                        "Risk": round(h.get("risk", 0) * 100, 1),
                        "Depression": round(h.get("depression", 0) * 100, 1),
                        "Anxiety": round(h.get("anxiety", 0) * 100, 1),
                        "Stress": round(h.get("stress", 0) * 100, 1),
                    })
                
                df = pd.DataFrame(data).set_index("Time")
                
                st.markdown("**Overall Risk Over Time**")
                st.line_chart(df[["Risk"]], use_container_width=True)

                st.markdown("**Detailed Metrics Over Time**")
                st.line_chart(df[["Depression", "Anxiety", "Stress"]], use_container_width=True)

                st.markdown("---")
                st.markdown('<div class="section-label">Session History (last 7)</div>',
                            unsafe_allow_html=True)
                for h in reversed(history):
                    ts     = fmt_ts(h.get("timestamp", ""))
                    risk   = round(h.get("risk", 0) * 100, 1)
                    dep    = round(h.get("depression", 0) * 100, 1)
                    anx    = round(h.get("anxiety", 0) * 100, 1)
                    stress = round(h.get("stress", 0) * 100, 1)
                    pill   = risk_pill(risk)
                    st.markdown(f'''
                    <div class="hist-row">
                        <span class="hist-ts">{ts}</span>
                        <span>Depression <b>{dep}</b></span>
                        <span>Anxiety <b>{anx}</b></span>
                        <span>Stress <b>{stress}</b></span>
                        <span>Risk <b>{risk}</b> {pill}</span>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("No history yet — run a fusion analysis to start recording sessions.")

# ─────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────
def main():
    if not st.session_state.token:
        page_auth()
        return

    page = sidebar()

    if "Fusion" in page:
        page_fusion()
    elif "Dashboard" in page:
        page_dashboard()

if __name__ == "__main__":
    main()
