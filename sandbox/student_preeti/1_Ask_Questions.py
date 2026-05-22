"""
EmpathBot — Q&A mode.

The user asks questions in plain words about AI in their healthcare.
Webcam runs in the background; if sustained distress is detected,
the bot offers to re-explain more simply.

Currently UI-only — the LLM call is stubbed via `mock_bot_reply`.
Wire `utils.llm_router.get_reply` in when ready.
"""
import streamlit as st
from utils.theme import apply_theme
from utils.session import init_session
from utils.bot_stub import mock_bot_reply

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ask EmpathBot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)
apply_theme()
init_session()

if not st.session_state.get("consent_given"):
    st.warning("Please return to the home page and tick the consent box first.")
    if st.button("← Back to start"):
        st.switch_page("Home.py")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
col_back, col_title = st.columns([1, 4])
with col_back:
    if st.button("← Home", key="home_btn"):
        st.switch_page("Home.py")
with col_title:
    st.markdown("## Ask me anything")

st.markdown(
    "Type your question below, or tap the **microphone** to speak. "
    "I'll do my best to answer in plain words."
)

# ── Suggested starter questions ───────────────────────────────────────────────
if not st.session_state["chat_history"]:
    st.markdown("#### Some things people often ask:")
    suggestions = [
        "What does AI actually do with my scan?",
        "Will a real doctor still look at my results?",
        "Is my data safe?",
        "Why is the GP using a computer to help decide?",
    ]
    sug_cols = st.columns(2)
    for i, s in enumerate(suggestions):
        with sug_cols[i % 2]:
            if st.button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state["chat_history"].append(
                    {"role": "user", "text": s, "emotion": None}
                )
                reply = mock_bot_reply(s)
                st.session_state["chat_history"].append(
                    {"role": "bot", "text": reply, "emotion": None}
                )
                st.rerun()

# ── Chat transcript ───────────────────────────────────────────────────────────
st.markdown("<div class='eb-spacer'></div>", unsafe_allow_html=True)

for msg in st.session_state["chat_history"]:
    cls = "eb-msg-user" if msg["role"] == "user" else "eb-msg-bot"
    st.markdown(f"<div class='{cls}'>{msg['text']}</div>", unsafe_allow_html=True)

# ── Input row ─────────────────────────────────────────────────────────────────
st.markdown("<div class='eb-spacer'></div>", unsafe_allow_html=True)

with st.form("ask_form", clear_on_submit=True):
    user_text = st.text_area(
        "Your question",
        placeholder="Type here... e.g. 'Is the computer making decisions about my care?'",
        height=100,
        label_visibility="collapsed",
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        mic = st.form_submit_button("🎤 Speak instead", use_container_width=True)
    with c2:
        send = st.form_submit_button(
            "Send question", type="primary", use_container_width=True
        )

if mic:
    st.info("Microphone input coming soon — please type for now.")
    # TODO: integrate browser Web Speech API via streamlit-mic-recorder

if send and user_text.strip():
    # Pull latest dominant emotion from buffer (set by webcam page/component)
    current_emotion = (
        max(set(st.session_state["emotion_window"]),
            key=st.session_state["emotion_window"].count)
        if st.session_state["emotion_window"] else None
    )

    st.session_state["chat_history"].append(
        {"role": "user", "text": user_text, "emotion": current_emotion}
    )
    reply = mock_bot_reply(user_text, emotion=current_emotion)
    st.session_state["chat_history"].append(
        {"role": "bot", "text": reply, "emotion": None}
    )
    st.rerun()

# ── End session ───────────────────────────────────────────────────────────────
st.markdown("<div class='eb-spacer'></div>", unsafe_allow_html=True)
if st.session_state["chat_history"]:
    if st.button("I'm done — finish session", use_container_width=True):
        st.session_state["chat_history"] = []
        st.switch_page("Home.py")
