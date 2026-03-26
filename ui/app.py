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

# ── Generate a session ID once per browser session ───────────
# This persists across reruns but resets on page refresh
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ── Header ────────────────────────────────────────────────────
st.title("🥗 Food Suitability Advisor")
st.caption("AI-powered dietary guidance using RAG + ML + Groq (Llama 3.1 8B)")
# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Try an example")
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
    st.subheader("System status")
    try:
        r = requests.get("http://localhost:8000/health", timeout=2)
        if r.status_code == 200:
            st.success("API running")
        else:
            st.error("API error")
    except:
        st.error("API offline — start uvicorn")

    st.divider()
    st.caption(
        "This tool provides general dietary information only. "
        "Always consult a qualified healthcare professional "
        "before making dietary changes."
    )

# ── Chat history ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant" and msg.get("ml_label"):
            label = msg["ml_label"]
            conf  = msg["ml_confidence"]
            color = {"safe": "green", "moderate": "orange", "avoid": "red"}.get(label, "gray")
            st.markdown(f"**ML Verdict:** :{color}[{label.upper()}] — {conf}% confidence")

        if msg["role"] == "assistant" and msg.get("entities"):
            ents = msg["entities"]
            with st.expander("What the system detected"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Age",       ents.get('age', 'N/A'))
                col2.metric("Condition", (ents.get('condition') or 'N/A').replace('_', ' ').title())
                col3.metric("Food",      (ents.get('food') or 'N/A').title())

        if msg["role"] == "assistant" and msg.get("shap_reasons"):
            with st.expander("Why this verdict"):
                for feat, val in msg["shap_reasons"]:
                    icon = "🔴" if val > 0 else "🟢"
                    direction = "risk" if val > 0 else "protective"
                    st.write(f"{icon} **{feat.title()}** — {direction} (SHAP: {val:+.3f})")

# ── Input ─────────────────────────────────────────────────────
prefill = st.session_state.pop('prefill', '')
query   = st.chat_input("Ask about a food — e.g. I am 64, diabetic. Can I eat ice cream?")

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

                    st.write(text)

                    # Only show ML details on a full pipeline response
                    if response_type == "response":
                        label = data.get("ml_label", "")
                        conf  = data.get("ml_confidence", 0)
                        color = {"safe": "green", "moderate": "orange", "avoid": "red"}.get(label, "gray")
                        st.markdown(f"**ML Verdict:** :{color}[{label.upper()}] — {conf}% confidence")

                        ents = data.get("entities") or {}
                        if ents:
                            with st.expander("What the system detected"):
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Age",       ents.get('age', 'N/A'))
                                col2.metric("Condition", (ents.get('condition') or 'N/A').replace('_', ' ').title())
                                col3.metric("Food",      (ents.get('food') or 'N/A').title())

                        shap = data.get("shap_reasons") or []
                        if shap:
                            with st.expander("Why this verdict"):
                                for feat, val in shap:
                                    icon = "🔴" if val > 0 else "🟢"
                                    direction = "risk" if val > 0 else "protective"
                                    st.write(f"{icon} **{feat.title()}** — {direction} (SHAP: {val:+.3f})")

                        st.session_state.messages.append({
                            "role":          "assistant",
                            "content":       text,
                            "ml_label":      label,
                            "ml_confidence": conf,
                            "entities":      ents,
                            "shap_reasons":  shap,
                        })

                    else:
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