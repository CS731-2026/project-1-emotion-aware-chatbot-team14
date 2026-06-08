"""
EmpathBot, Feedback mode.

Walks the user through 4 short questions about their visit. The webcam runs
in the background; emotion labels are collected per-question via the rolling
buffer in session_state["emotion_window"].

After each answer:
  - stated sentiment is classified (VADER stub here)
  - facial emotion ratio over the answer window is computed
  - mismatch flag stored (NEVER shown to the patient)

At the end, results are appended to data/sessions.jsonl for the staff dashboard.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from utils.theme import apply_theme
from utils.session import init_session
from utils.mismatch_stub import detect_mismatch
from utils.webcam_stub import webcam_panel

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Give Feedback",
    page_icon="📝",
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

QUESTIONS = [
    "How would you say your visit went today?",
    "Did the doctor explain what the AI was used for in a way you understood?",
    "How did you feel about a computer being involved in your care?",
    "Is there anything you're still worried about?",
]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)
SESSIONS_FILE = DATA_DIR / "sessions.jsonl"

step = st.session_state["feedback_step"]
total = len(QUESTIONS)

# ── Header / progress ─────────────────────────────────────────────────────────
col_back, col_prog = st.columns([1, 4])
with col_back:
    if st.button("← Home", key="home_btn"):
        st.switch_page("Home.py")
with col_prog:
    if step < total:
        st.markdown(f"**Question {step + 1} of {total}**")
        st.progress((step) / total)

# ── Webcam panel (small, top-right feel) ──────────────────────────────────────
with st.expander("Your webcam (so I can see how you're feeling)", expanded=False):
    webcam_panel()  # stub, will plug ResNet18 inferencer here

# ── Step: ask question or finish ──────────────────────────────────────────────
if step < total:
    st.markdown(f"### {QUESTIONS[step]}")

    with st.form(f"q_form_{step}", clear_on_submit=True):
        answer = st.text_area(
            "Your answer",
            placeholder="Take your time. There's no right or wrong answer.",
            height=140,
            label_visibility="collapsed",
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            skip = st.form_submit_button("Skip this one", use_container_width=True)
        with c2:
            submit = st.form_submit_button(
                "Next →", type="primary", use_container_width=True
            )

    if submit and answer.strip():
        # Snapshot emotion window for this answer
        emotion_window = list(st.session_state["emotion_window"])
        result = detect_mismatch(answer, emotion_window)

        st.session_state["feedback_answers"].append({
            "q": QUESTIONS[step],
            "a": answer.strip(),
            "stated_sentiment": result["stated_sentiment"],
            "facial_negativity": result["facial_negativity"],
            "dominant_facial_emotion": result["dominant_facial_emotion"],
            "mismatch": result["mismatch"],
        })
        # Clear window for next question
        st.session_state["emotion_window"] = []
        st.session_state["feedback_step"] += 1
        st.rerun()

    if skip:
        st.session_state["feedback_answers"].append({
            "q": QUESTIONS[step],
            "a": "[skipped]",
            "stated_sentiment": None,
            "facial_negativity": None,
            "dominant_facial_emotion": None,
            "mismatch": False,
        })
        st.session_state["emotion_window"] = []
        st.session_state["feedback_step"] += 1
        st.rerun()

else:
    # ── Done ──────────────────────────────────────────────────────────────────
    st.markdown("## Thank you")
    st.markdown(
        "Your answers help the clinic understand how patients feel about "
        "AI being used in their care. **A staff member will review the summary later**, "
        "your individual answers stay private."
    )

    # Persist session for the staff dashboard
    record = {
        "session_id": st.session_state["session_id"],
        "started": st.session_state["session_started"],
        "finished": datetime.now().isoformat(timespec="seconds"),
        "answers": st.session_state["feedback_answers"],
    }
    with SESSIONS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")

    st.success("Feedback saved. You can close this window now.")

    if st.button("Start over", use_container_width=True):
        st.session_state["feedback_step"] = 0
        st.session_state["feedback_answers"] = []
        st.switch_page("Home.py")
