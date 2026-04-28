import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000/chat"

st.set_page_config(
    page_title="Food Suitability Advisor",
    page_icon="🥗",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    :root {
        --bg:            #f8fafc;
        --bg-soft:       #f1f5f9;
        --paper:         #ffffff;
        --paper-strong:  #ffffff;
        --ink:           #0f172a;
        --muted:         #0f172a;
        --line:          rgba(100, 116, 139, 0.12);
        --shadow-sm:     0 1px 2px rgba(0, 0, 0, 0.03), 0 1px 1px rgba(0, 0, 0, 0.02);
        --shadow:        0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.02);
        --shadow-lg:     0 20px 40px -12px rgba(0, 0, 0, 0.1);
        --blue-50:       #eff6ff;
        --blue-100:      #dbeafe;
        --blue-200:      #bfdbfe;
        --blue-500:      #3b82f6;
        --blue-600:      #2563eb;
        --blue-700:      #1d4ed8;
        --safe-bg:       #f0fdf4;
        --safe-ink:      #166534;
        --safe-border:   rgba(34, 197, 94, 0.25);
        --safe-dot:      #22c55e;
        --mod-bg:        #fffbeb;
        --mod-ink:       #b45309;
        --mod-border:    rgba(245, 158, 11, 0.25);
        --mod-dot:       #f59e0b;
        --avoid-bg:      #fef2f2;
        --avoid-ink:     #b91c1c;
        --avoid-border:  rgba(239, 68, 68, 0.25);
        --avoid-dot:     #ef4444;
        --radius-sm:     10px;
        --radius-md:     16px;
        --radius-lg:     22px;
        --radius-xl:     28px;
    }

    *, *::before, *::after {
        font-family: 'Plus Jakarta Sans', sans-serif;
        box-sizing: border-box;
    }

    /* App background */
    .stApp {
        background: linear-gradient(155deg, #eaf1f9 0%, #f0f5fb 40%, #f5f0f9 100%);
        color: var(--ink);
    }
    [data-testid="stAppViewContainer"] { background: transparent; }

    /* Sidebar — quiet, de-emphasised */
    [data-testid="stSidebar"] {
        background: #f7f9fc;
        border-right: 1px solid var(--line);
        box-shadow: none;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.6rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    [data-testid="stSidebar"] * {
        font-size: 0.84rem !important;
        color: var(--muted) !important;
    }
    [data-testid="stSidebar"] h3 {
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700 !important;
        color: #a0b0c4 !important;
        margin-bottom: 0.5rem;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        border: 1px solid rgba(99, 132, 168, 0.18) !important;
        border-radius: 8px !important;
        color: var(--muted) !important;
        font-size: 0.8rem !important;
        padding: 0.35rem 0.6rem !important;
        font-weight: 400 !important;
        box-shadow: none !important;
        transition: border-color 0.15s, color 0.15s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: var(--blue-500) !important;
        color: var(--blue-600) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(99, 132, 168, 0.12) !important;
        margin: 0.75rem 0 !important;
    }
    [data-testid="stSidebar"] .stSuccess,
    [data-testid="stSidebar"] .stError {
        font-size: 0.75rem !important;
        padding: 0.3rem 0.6rem !important;
        border-radius: 6px !important;
        opacity: 0.65;
    }

    /* Main block */
    .block-container {
        max-width: 820px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hero banner */
    .hero-shell {
        position: relative;
        overflow: hidden;
        border-radius: var(--radius-xl);
        padding: 2rem 2.2rem 1.8rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #ffffff 0%, #f0f6ff 60%, #e8f0fb 100%);
        border: 1px solid rgba(99, 149, 220, 0.18);
        box-shadow: var(--shadow-lg);
    }
    .hero-shell:before {
        content: "";
        position: absolute;
        top: -80px; right: -60px;
        width: 260px; height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(59, 130, 196, 0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-shell:after {
        content: "";
        position: absolute;
        bottom: -70px; left: -50px;
        width: 220px; height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.07) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-kicker {
        position: relative; z-index: 1;
        display: inline-flex; align-items: center; gap: 0.4rem;
        margin-bottom: 0.75rem;
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        background: var(--blue-50);
        border: 1px solid var(--blue-200);
        color: var(--blue-600);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .hero-kicker:before {
        content: "";
        display: inline-block;
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--blue-500);
    }
    .hero-title {
        position: relative; z-index: 1;
        font-size: 1.9rem;
        line-height: 1.15;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero-copy {
        position: relative; z-index: 1;
        max-width: 580px;
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.65;
        font-weight: 400;
    }
    .hero-grid {
        position: relative; z-index: 1;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 1.4rem;
    }
    .hero-stat {
        padding: 0.9rem 1rem;
        border-radius: var(--radius-md);
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(99, 149, 220, 0.14);
        backdrop-filter: blur(8px);
        box-shadow: var(--shadow-sm);
    }
    .hero-stat-label {
        color: var(--muted);
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .hero-stat-value {
        color: var(--ink);
        font-size: 0.88rem;
        font-weight: 600;
        line-height: 1.35;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 0.25rem 0 !important;
    }
    [data-testid="stChatMessageContent"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    /* Result cards */
    .result-card {
        border-radius: var(--radius-xl);
        padding: 1.6rem 1.7rem 1.4rem;
        margin: 0.5rem 0 1rem;
        background: var(--paper);
        border: 1px solid var(--line);
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    .result-card:before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: var(--radius-xl) var(--radius-xl) 0 0;
    }
    .result-card.safe { border-color: var(--safe-border); }
    .result-card.safe:before { background: linear-gradient(90deg, #22c55e, #86efac); }
    .result-card.moderate { border-color: var(--mod-border); }
    .result-card.moderate:before { background: linear-gradient(90deg, #f59e0b, #fcd34d); }
    .result-card.avoid { border-color: var(--avoid-border); }
    .result-card.avoid:before { background: linear-gradient(90deg, #ef4444, #fca5a5); }

    .result-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.38rem 0.85rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-bottom: 0.9rem;
        text-transform: uppercase;
    }
    .result-badge:before {
        content: "";
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
    }
    .badge-safe { background: var(--safe-bg); color: var(--safe-ink); border: 1px solid var(--safe-border); }
    .badge-safe:before { background: var(--safe-dot); }
    .badge-moderate { background: var(--mod-bg); color: var(--mod-ink); border: 1px solid var(--mod-border); }
    .badge-moderate:before { background: var(--mod-dot); }
    .badge-avoid { background: var(--avoid-bg); color: var(--avoid-ink); border: 1px solid var(--avoid-border); }
    .badge-avoid:before { background: var(--avoid-dot); }

    .result-title {
        font-size: 1.22rem;
        line-height: 1.4;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 0.6rem;
        max-width: 640px;
        letter-spacing: -0.01em;
    }
    .result-subrow {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.6rem;
    }
    .result-confidence { color: var(--muted); font-size: 0.85rem; font-weight: 400; }
    .confidence-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: var(--blue-50);
        border: 1px solid var(--blue-200);
        color: var(--blue-600);
        font-size: 0.75rem;
        font-weight: 700;
    }

    /* Section cards */
    .section-card {
        border: 1px solid var(--line);
        border-radius: var(--radius-lg);
        padding: 1.1rem 1.2rem;
        margin-top: 0.9rem;
        background: var(--paper-strong);
        box-shadow: var(--shadow-sm);
    }
    .section-label {
        color: var(--blue-500);
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .section-copy { color: var(--ink); line-height: 1.7; font-size: 0.96rem; font-weight: 400; }
    .section-copy p { margin: 0 0 0.5rem 0; }
    .section-copy p:last-child { margin-bottom: 0; }

    /* Factors */
    .factor-item {
        display: flex;
        align-items: flex-start;
        gap: 0.65rem;
        padding: 0.6rem 0;
        border-top: 1px solid rgba(99, 132, 168, 0.09);
    }
    .factor-item:first-child { border-top: 0; padding-top: 0; }
    .factor-dot { width: 10px; height: 10px; border-radius: 50%; margin-top: 0.4rem; flex: 0 0 10px; }
    .factor-risk { background: #ef4444; box-shadow: 0 0 0 4px rgba(239, 68, 68, 0.15); }
    .factor-protective { background: #22c55e; box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.15); }
    .factor-text { color: #374a5e; line-height: 1.55; font-size: 0.93rem; }

    /* Detail grid */
    .detail-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; }
    .detail-card {
        border-radius: var(--radius-md);
        padding: 0.85rem 0.95rem;
        background: var(--blue-50);
        border: 1px solid var(--blue-200);
    }
    .detail-label { color: var(--muted); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600; margin-bottom: 0.3rem; }
    .detail-value { color: var(--ink); font-size: 1rem; font-weight: 700; line-height: 1.25; word-break: break-word; }

    /* Source chips */
    .source-chip {
        display: inline-block;
        padding: 0.32rem 0.7rem;
        margin: 0.25rem 0.3rem 0 0;
        border-radius: 999px;
        background: var(--blue-50);
        border: 1px solid var(--blue-200);
        color: var(--blue-600);
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Disclaimer */
    .micro-note {
        color: var(--muted);
        font-size: 0.83rem;
        line-height: 1.55;
        margin-top: 0.9rem;
        padding: 0.75rem 1rem;
        background: var(--blue-50);
        border-radius: var(--radius-sm);
        border-left: 3px solid var(--blue-200);
    }

    /* Expander */
    [data-testid="stExpander"] {
        border: 1px solid var(--line) !important;
        border-radius: var(--radius-md) !important;
        background: var(--paper-strong) !important;
        box-shadow: var(--shadow-sm) !important;
        margin-top: 0.9rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: var(--radius-md);
        border: 1px solid var(--line);
        background: var(--paper);
        color: var(--ink);
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: var(--blue-500);
        color: var(--blue-600);
        box-shadow: var(--shadow);
    }

    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    hr { border-color: var(--line) !important; }

    @media (max-width: 760px) {
        .hero-grid, .detail-grid { grid-template-columns: 1fr; }
        .hero-title { font-size: 1.45rem; }
        .result-title { font-size: 1.05rem; }
        .result-card { padding: 1.2rem 1.1rem; }
        .hero-shell { padding: 1.4rem 1.2rem; }
    }
</style>
""", unsafe_allow_html=True)


def _parse_response_sections(text: str) -> dict:
    sections = {
        "recommendation": "",
        "body": [],
        "top_factors": [],
        "evidence": [],
        "confidence": "",
        "disclaimer": "",
    }
    current = "body"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("Recommendation:"):
            sections["recommendation"] = line.replace("Recommendation:", "", 1).strip()
            current = None
        elif line == "Top factors:":
            current = "top_factors"
        elif line.startswith("- ") and current == "top_factors":
            sections["top_factors"].append(line[2:].strip())
        elif line.startswith("Evidence:"):
            value = line.replace("Evidence:", "", 1).strip()
            sections["evidence"] = [item.strip() for item in value.split(";") if item.strip()]
            current = None
        elif line.startswith("Confidence:"):
            sections["confidence"] = line
            current = None
        elif line.startswith("Please consult your doctor"):
            sections["disclaimer"] = line
            current = None
        else:
            sections["body"].append(line)

    return sections


def _render_assistant_response(msg: dict):
    if msg["role"] != "assistant" or not msg.get("ml_label"):
        st.write(msg["content"])
        return

    sections = _parse_response_sections(msg["content"])
    label = msg["ml_label"]
    conf = msg["ml_confidence"]
    badge_class = {
        "safe": "badge-safe",
        "moderate": "badge-moderate",
        "avoid": "badge-avoid",
    }.get(label, "badge-moderate")
    title = {
        "safe": "Generally okay",
        "moderate": "Moderation advised",
        "avoid": "Best avoided",
    }.get(label, label.title())
    card_class = {
        "safe": "safe",
        "moderate": "moderate",
        "avoid": "avoid",
    }.get(label, "moderate")

    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="result-badge {badge_class}">{title}</div>
            <div class="result-title">{sections['recommendation'] or 'Dietary guidance available below.'}</div>
            <div class="result-subrow">
                <div class="result-confidence">Tailored food guidance from nutrition signals, retrieval evidence, and model reasoning.</div>
                <div class="confidence-pill">Confidence {conf}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if sections["body"]:
        body_html = "".join([f"<p>{paragraph}</p>" for paragraph in sections["body"]])
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-label">Why</div>
                <div class="section-copy">{body_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if sections["top_factors"]:
        factor_blocks = []
        for item in sections["top_factors"]:
            css_class = "factor-risk" if "increase" in item.lower() or "concern" in item.lower() else "factor-protective"
            factor_blocks.append(
                f"""
                <div class="factor-item">
                    <div class="factor-dot {css_class}"></div>
                    <div class="factor-text">{item}</div>
                </div>
                """
            )
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-label">Top factors</div>
                {''.join(factor_blocks)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if sections["evidence"]:
        chips = "".join([f'<span class="source-chip">{source}</span>' for source in sections["evidence"]])
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-label">Evidence</div>
                <div>{chips}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    rag = msg.get("rag_results") or []
    if rag:
        with st.expander("📄 Retrieved Evidence", expanded=False):
            for i, chunk in enumerate(rag):
                st.write(chunk)
                score = chunk.get("score", 0)
                text  = chunk.get("text", "").strip()
                src   = chunk.get("source", "Unknown source")

                # Score bar colour — green if strong, yellow if moderate
                bar_color = (
                    "#22c55e" if score >= 0.75 else
                    "#f59e0b" if score >= 0.55 else
                    "#94a3b8"
                )

                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid rgba(99,149,220,0.18);
                        border-radius: 12px;
                        padding: 0.9rem 1rem;
                        margin-bottom: 0.75rem;
                        background: #f8fafc;
                    ">
                        <div style="display:flex; justify-content:space-between; 
                                    align-items:center; margin-bottom:0.5rem;">
                            <span style="
                                font-size:0.7rem; font-weight:700;
                                text-transform:uppercase; letter-spacing:0.08em;
                                color:#64748b;
                            ">Chunk {i+1} · {src}</span>
                            <span style="
                                background:{bar_color}22;
                                color:{bar_color};
                                border:1px solid {bar_color}55;
                                border-radius:999px;
                                padding:0.2rem 0.6rem;
                                font-size:0.72rem;
                                font-weight:700;
                            ">score {score:.3f}</span>
                        </div>
                        <div style="
                            font-size:0.9rem; line-height:1.65;
                            color:#334155;
                            border-left: 3px solid {bar_color};
                            padding-left: 0.75rem;
                        ">{text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    ents = msg.get("entities")
    if ents:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-label">Detected details</div>
                <div class="detail-grid">
                    <div class="detail-card">
                        <div class="detail-label">Age</div>
                        <div class="detail-value">{ents.get("age", "N/A")}</div>
                    </div>
                    <div class="detail-card">
                        <div class="detail-label">Condition</div>
                        <div class="detail-value">{(ents.get("condition") or "N/A").replace("_", " ").title()}</div>
                    </div>
                    <div class="detail-card">
                        <div class="detail-label">Food</div>
                        <div class="detail-value">{(ents.get("food") or "N/A").title()}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    shap = msg.get("shap_reasons")
    if shap:
        with st.expander("Model factors"):
            for feat, val in shap:
                icon = "🔴" if val > 0 else "🟢"
                direction = "risk" if val > 0 else "protective"
                st.write(f"{icon} **{feat.title()}** — {direction} (SHAP: {val:+.3f})")

    if sections["disclaimer"]:
        st.markdown(f'<div class="micro-note">{sections["disclaimer"]}</div>', unsafe_allow_html=True)

# ── Generate a session ID once per browser session ───────────
# This persists across reruns but resets on page refresh
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ── Header ────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-kicker">Food Health Assistant</div>
        <div class="hero-title">Understand whether a food fits your health needs.</div>
        <div class="hero-copy">
            Ask about any meal or ingredient and get a clearer recommendation, the nutrition logic behind it,
            and supporting dietary evidence in one place.
        </div>
        <div class="hero-grid">
            <div class="hero-stat">
                <div class="hero-stat-label">Looks at</div>
                <div class="hero-stat-value">nutrition, condition, and retrieval evidence</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-label">Best for</div>
                <div class="hero-stat-value">diabetes, hypertension, kidney disease, and general guidance</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-label">Tone</div>
                <div class="hero-stat-value">short, practical, and easier to trust</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Try an example")
    examples = [
        "I am a 60 year old male with diabetes. I have type 2 diabetes and a sedentary lifestyle. Can I consume beverages?",
        "I am a 58 year old female with hypertension. I take diuretics and have low activity levels. Is it safe for me to eat salted potato chips?",
        "I am a 50 year old male with kidney disease stage 4 and I am on medication. Can I eat blueberries?",
        "I am 73 year old male, with height 81kg and height 158cm. i have kidney disease stage 3. i am a lightly active person. Can i eat pork?",
        "I am a 35 year old female with hypertension and I follow a moderately active lifestyle. Is it safe to eat spinach?",
        "I am a 40 year old male with diabetes. I am physically active and maintain a normal BMI. Can I eat pineapple?",
        "I am a 45 year old male with diabetes. I am moderately active. Can I eat whole pasta?",
        "I am a 30 year old healthy male. Can I eat Pineapple?"

    ]
    for ex in examples:
        if st.button(ex[:42] + "...", use_container_width=True):
            st.session_state['prefill'] = ex

    st.divider()

    # Session info
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

    # Reset button
    if st.button("Start new conversation", use_container_width=True):
        try:
            requests.post(
                f"http://localhost:8000/reset/{st.session_state.session_id}",
                timeout=5
            )
        except:
            pass
        st.session_state.session_id  = str(uuid.uuid4())
        st.session_state.messages    = []
        st.rerun()

    st.divider()

    # API status
    st.markdown("### System status")
    try:
        r = requests.get("http://localhost:8000/health", timeout=2)
        if r.status_code == 200:
            st.success("API running")
        else:
            st.error("API error")
    except:
        st.error("API offline — start uvicorn")

    st.divider()
    st.markdown(
        """
        <div class="micro-note">
            This tool gives general dietary guidance only. It should support decision-making, not replace
            personal advice from a qualified clinician or dietitian.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Chat history ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        _render_assistant_response(msg)

# ── Input ─────────────────────────────────────────────────────
prefill = st.session_state.pop('prefill', '')
query   = st.chat_input("Ask about a food, meal, or ingredient.")

if prefill and not query:
    query = prefill

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Call API with session_id + message
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                res = requests.post(
                    API_URL,
                    json={
                        "session_id": st.session_state.session_id,
                        "message":    query
                    },
                    timeout=120
                )

                if res.status_code == 200:
                    data = res.json()
                    response_type = data.get("type")
                    text          = data.get("text", "")

                    # Only show ML details on a full pipeline response
                    if response_type == "response":
                        label = data.get("ml_label", "")
                        conf  = data.get("ml_confidence", 0)
                        ents = data.get("entities") or {}
                        shap = data.get("shap_reasons") or []
                        rag   = data.get("rag_results") or []

                        assistant_message = {
                            "role":          "assistant",
                            "content":       text,
                            "ml_label":      label,
                            "ml_confidence": conf,
                            "entities":      ents,
                            "shap_reasons":  shap,
                            "rag_results":   rag,
                        }
                        _render_assistant_response(assistant_message)
                        st.session_state.messages.append(assistant_message)

                    else:
                        st.write(text)
                        # Conversational message — asking for missing field
                        st.session_state.messages.append({
                            "role":    "assistant",
                            "content": text,
                        })

                else:
                    err = res.json().get("detail", res.text)
                    st.error(f"API error {res.status_code}: {err}")

            except requests.exceptions.Timeout:
                st.error("Request timed out — please try again.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure uvicorn is running.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")