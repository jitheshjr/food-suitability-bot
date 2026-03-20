import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.set_page_config(
    page_title="Food Suitability Advisor",
    page_icon="🥗",
    layout="centered"
)

# ── Header ────────────────────────────────────────
st.title("🥗 Food Suitability Advisor")
st.caption("AI-powered dietary guidance using RAG + ML + TinyLlama")

# ── Sidebar ───────────────────────────────────────
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

    # Show system status
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

# ── Chat history ──────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Show verdict badge for assistant messages
        if msg["role"] == "assistant" and msg.get("ml_label"):
            label = msg["ml_label"]
            conf  = msg["ml_confidence"]
            color = {"safe": "green", "moderate": "orange", "avoid": "red"}.get(label, "gray")
            st.markdown(f"**ML Verdict:** :{color}[{label.upper()}] — {conf}% confidence")

        # Show detected entities
        if msg["role"] == "assistant" and msg.get("entities"):
            ents = msg["entities"]
            with st.expander("What the system detected"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Age",       ents.get('age', 'N/A'))
                col2.metric("Condition", (ents.get('condition') or 'N/A').replace('_',' ').title())
                col3.metric("Food",      (ents.get('food') or 'N/A').title())

        # Show SHAP reasons
        if msg["role"] == "assistant" and msg.get("shap_reasons"):
            with st.expander("Why this verdict"):
                for feat, val in msg["shap_reasons"]:
                    direction = "risk" if val > 0 else "protective"
                    bar_color = "🔴" if val > 0 else "🟢"
                    st.write(f"{bar_color} **{feat.title()}** — {direction} factor (SHAP: {val:+.3f})")

# ── Input ─────────────────────────────────────────
prefill = st.session_state.pop('prefill', '')
query   = st.chat_input("e.g. I am a 64 year old diabetic, can I eat ice cream?")

if prefill and not query:
    query = prefill

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Call API and show response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing — this may take 20–30 seconds..."):
            try:
                res = requests.post(
                    API_URL,
                    json={"query": query},
                    timeout=120
                )

                if res.status_code == 200:
                    data = res.json()

                    # Main response
                    st.write(data["response"])

                    # Verdict badge
                    label = data.get("ml_label", "")
                    conf  = data.get("ml_confidence", 0)
                    color = {"safe":"green","moderate":"orange","avoid":"red"}.get(label,"gray")
                    st.markdown(f"**ML Verdict:** :{color}[{label.upper()}] — {conf}% confidence")

                    # Detected entities
                    ents = data.get("entities", {})
                    with st.expander("What the system detected"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Age",       ents.get('age', 'N/A'))
                        col2.metric("Condition", (ents.get('condition') or 'N/A').replace('_',' ').title())
                        col3.metric("Food",      (ents.get('food') or 'N/A').title())

                    # SHAP explanation
                    shap = data.get("shap_reasons", [])
                    if shap:
                        with st.expander("Why this verdict"):
                            for feat, val in shap:
                                direction = "risk" if val > 0 else "protective"
                                icon = "🔴" if val > 0 else "🟢"
                                st.write(f"{icon} **{feat.title()}** — {direction} (SHAP: {val:+.3f})")

                    # Save to history
                    st.session_state.messages.append({
                        "role":         "assistant",
                        "content":      data["response"],
                        "ml_label":     label,
                        "ml_confidence": conf,
                        "entities":     ents,
                        "shap_reasons": shap,
                    })

                else:
                    st.error(f"API error {res.status_code}: {res.text}")

            except requests.exceptions.Timeout:
                st.error("Request timed out. TinyLlama is still processing — please try again.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure uvicorn is running in another terminal.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
