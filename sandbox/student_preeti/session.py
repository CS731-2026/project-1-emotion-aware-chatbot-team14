"""
Session state helpers for EmpathBot.

Streamlit reruns the script on every interaction, so all cross-rerun state
lives in st.session_state. This module centralises initialisation so every
page sees the same keys.
"""
import uuid
from datetime import datetime
import streamlit as st


def init_session() -> None:
    """Initialise session state keys if not already present."""
    defaults = {
        "session_id": str(uuid.uuid4())[:8],
        "session_started": datetime.now().isoformat(timespec="seconds"),
        "consent_given": False,
        # Q&A mode
        "chat_history": [],            # list[{"role": "user"|"bot", "text": str, "emotion": str|None}]
        # Feedback mode
        "feedback_step": 0,            # current question index
        "feedback_answers": [],        # list[{"q": str, "a": str, "stated": str, "facial": dict, "mismatch": bool}]
        # Emotion buffer (rolling) — populated by the webcam callback
        "emotion_window": [],          # recent emotion labels
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
