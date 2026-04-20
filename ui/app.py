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
    :root {
        --bg: #ffffff;
        --bg-soft: #ffffff;
        --paper: rgba(255, 252, 247, 0.92);
        --paper-strong: #ffffff;
        --ink: #223127;
        --muted: #6d756b;
        --line: rgba(62, 73, 64, 0.14);
        --shadow: 0 16px 40px rgba(88, 67, 44, 0.08);
        --safe-bg: #dff2dd;
        --safe-ink: #285c34;
        --safe-accent: #80af79;
        --mod-bg: #ffffff;
        --mod-ink: #8b5412;
        --mod-accent: #d8a15d;
        --avoid-bg: #f7d8d1;
        --avoid-ink: #98382d;
        --avoid-accent: #cc7b6d;
        --sage: #dbe7d3;
        --moss: #4d684e;
        --cream: #f3ece1;
        --gold: #d7b173;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(215, 177, 115, 0.16), transparent 28%),
            radial-gradient(circle at top right, rgba(129, 167, 122, 0.16), transparent 24%),
            linear-gradient(180deg, var(--bg-soft) 0%, var(--bg) 100%);
        color: var(--ink);
    }
    [data-testid="stAppViewContainer"] {
        background: transparent;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f2eadf 0%, #ede3d6 100%);
        border-right: 1px solid rgba(83, 72, 55, 0.09);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    .block-container {
        max-width: 880px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.02em;
    }
    .hero-shell {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(70, 83, 71, 0.12);
        border-radius: 28px;
        padding: 1.4rem 1.45rem 1.15rem 1.45rem;
        margin-bottom: 1.25rem;
        background:
            linear-gradient(135deg, rgba(255,250,243,0.95), rgba(244,236,225,0.92)),
            linear-gradient(180deg, rgba(255,255,255,0.55), rgba(255,255,255,0.15));
        box-shadow: var(--shadow);
    }
    .hero-shell:before {
        content: "";
        position: absolute;
        top: -55px;
        right: -45px;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(129, 167, 122, 0.28) 0%, rgba(129, 167, 122, 0.02) 70%);
    }
    .hero-shell:after {
        content: "";
        position: absolute;
        left: -38px;
        bottom: -60px;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(215, 177, 115, 0.24) 0%, rgba(215, 177, 115, 0.02) 68%);
    }
    .hero-kicker {
        position: relative;
        z-index: 1;
        display: inline-block;
        margin-bottom: 0.5rem;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        background: rgba(77, 104, 78, 0.1);
        color: var(--moss);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .hero-title {
        position: relative;
        z-index: 1;
        font-size: 2rem;
        line-height: 1.05;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .hero-copy {
        position: relative;
        z-index: 1;
        max-width: 640px;
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.55;
    }
    .hero-grid {
        position: relative;
        z-index: 1;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 1rem;
    }
    .hero-stat {
        padding: 0.82rem 0.9rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.55);
        border: 1px solid rgba(91, 98, 83, 0.1);
        backdrop-filter: blur(6px);
    }
    .hero-stat-label {
        color: var(--muted);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.22rem;
    }
    .hero-stat-value {
        color: var(--ink);
        font-size: 0.94rem;
        font-weight: 700;
    }
    .result-card {
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1.15rem 1.15rem 1rem 1.15rem;
        margin: 0.65rem 0 0.9rem 0;
        background: linear-gradient(180deg, rgba(255,250,244,0.96), rgba(248,242,233,0.98));
        box-shadow: var(--shadow);
    }
    .result-card.safe {
        box-shadow: inset 0 0 0 1px rgba(128, 175, 121, 0.18), var(--shadow);
    }
    .result-card.moderate {
        box-shadow: inset 0 0 0 1px rgba(216, 161, 93, 0.22), var(--shadow);
    }
    .result-card.avoid {
        box-shadow: inset 0 0 0 1px rgba(204, 123, 109, 0.22), var(--shadow);
    }
    .result-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.38rem;
        padding: 0.35rem 0.76rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin-bottom: 0.72rem;
        text-transform: uppercase;
    }
    .badge-safe {
        background: var(--safe-bg);
        color: var(--safe-ink);
    }
    .badge-moderate {
        background: var(--mod-bg);
        color: var(--mod-ink);
    }
    .badge-avoid {
        background: var(--avoid-bg);
        color: var(--avoid-ink);
    }
    .result-title {
        font-size: 1.18rem;
        line-height: 1.35;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 0.42rem;
        max-width: 680px;
    }
    .result-subrow {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.65rem;
    }
    .result-confidence {
        color: var(--muted);
        font-size: 0.92rem;
    }
    .confidence-pill {
        display: inline-block;
        padding: 0.28rem 0.56rem;
        border-radius: 999px;
        background: rgba(77, 104, 78, 0.08);
        color: var(--moss);
        font-size: 0.78rem;
        font-weight: 700;
    }
    .section-card {
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 0.95rem 1rem;
        margin-top: 0.8rem;
        background: rgba(255, 251, 245, 0.82);
        box-shadow: 0 10px 24px rgba(88, 67, 44, 0.04);
    }
    .section-label {
        color: var(--moss);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
        margin-bottom: 0.38rem;
    }
    .section-copy {
        color: var(--ink);
        line-height: 1.62;
        font-size: 0.98rem;
    }
    .factor-item {
        display: flex;
        align-items: flex-start;
        gap: 0.55rem;
        padding: 0.55rem 0;
        border-top: 1px dashed rgba(90, 96, 82, 0.14);
    }
    .factor-item:first-child {
        border-top: 0;
        padding-top: 0.1rem;
    }
    .factor-dot {
        width: 11px;
        height: 11px;
        border-radius: 50%;
        margin-top: 0.35rem;
        flex: 0 0 11px;
    }
    .factor-risk {
        background: #cd705b;
        box-shadow: 0 0 0 5px rgba(205, 112, 91, 0.14);
    }
    .factor-protective {
        background: #7ca871;
        box-shadow: 0 0 0 5px rgba(124, 168, 113, 0.14);
    }
    .factor-text {
        color: var(--ink);
        line-height: 1.45;
    }
    .detail-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.7rem;
    }
    .detail-card {
        border-radius: 18px;
        padding: 0.8rem 0.85rem;
        background: linear-gradient(180deg, rgba(245, 240, 231, 0.9), rgba(255, 250, 244, 0.95));
        border: 1px solid rgba(83, 72, 55, 0.08);
    }
    .detail-label {
        color: var(--muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.28rem;
    }
    .detail-value {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.2;
        word-break: break-word;
    }
    .source-chip {
        display: inline-block;
        padding: 0.33rem 0.66rem;
        margin: 0.24rem 0.35rem 0 0;
        border-radius: 999px;
        background: linear-gradient(180deg, #e7f0de, #dbe7d3);
        color: #35553b;
        font-size: 0.82rem;
        border: 1px solid rgba(77, 104, 78, 0.14);
        font-weight: 600;
    }
    .micro-note {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.45;
        margin-top: 0.85rem;
    }
    [data-testid="stChatMessage"] {
        background: transparent;
    }
    [data-testid="stChatMessageContent"] {
        padding-left: 0;
        padding-right: 0;
    }
    [data-testid="stExpander"] {
        border: 1px solid rgba(83, 72, 55, 0.08);
        border-radius: 18px;
        background: rgba(255, 251, 246, 0.68);
    }
    .stButton > button {
        border-radius: 14px;
        border: 1px solid rgba(77, 104, 78, 0.14);
        background: linear-gradient(180deg, #fffaf4, #f2eadf);
        color: var(--ink);
        font-weight: 600;
    }
    .stButton > button:hover {
        border-color: rgba(77, 104, 78, 0.24);
        color: var(--moss);
    }
    [data-testid="stMetricValue"] {
        color: var(--ink);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }
    @media (max-width: 760px) {
        .hero-grid,
        .detail-grid {
            grid-template-columns: 1fr;
        }
        .hero-title {
            font-size: 1.55rem;
        }
        .result-title {
            font-size: 1.05rem;
        }
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
        st.markdown(
            """
            <div class="section-card">
                <div class="section-label">Top factors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for item in sections["top_factors"]:
            icon = "🔴" if "increase" in item.lower() or "concern" in item.lower() else "🟢"
            st.write(f"{icon} {item}")

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
        "I am 64 years old diabetic. Can I eat ice cream?",
        "I am 55 with hypertension. Is it safe to eat chips?",
        "I am 45 with kidney disease. Can I eat chicken?",
        "I am 30 and healthy. Can I eat white rice daily?",
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

                        assistant_message = {
                            "role":          "assistant",
                            "content":       text,
                            "ml_label":      label,
                            "ml_confidence": conf,
                            "entities":      ents,
                            "shap_reasons":  shap,
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
