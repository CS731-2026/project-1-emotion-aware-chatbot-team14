"""
EmpathBot, Staff Dashboard.

Password-gated view for clinic staff. Shows:
  - Sessions today
  - Stated vs facial sentiment distribution
  - Flagged "polite but unhappy" sessions
  - Aggregate concerns
"""
import json
from collections import Counter
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.theme import apply_theme

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Staff Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_theme()

# Override the centered max-width for this page since it's a dashboard
st.markdown(
    "<style>.main .block-container { max-width: 1200px; }</style>",
    unsafe_allow_html=True,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SESSIONS_FILE = DATA_DIR / "sessions.jsonl"
STAFF_PASSWORD = "empath2026"  # TODO: env var in production

# ── Auth gate ─────────────────────────────────────────────────────────────────
if not st.session_state.get("staff_authed"):
    st.markdown("## Staff sign in")
    pw = st.text_input("Password", type="password")
    if st.button("Sign in", type="primary"):
        if pw == STAFF_PASSWORD:
            st.session_state["staff_authed"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
top_l, top_r = st.columns([4, 1])
with top_l:
    st.markdown("## Daily Patient Feedback Report")
    st.caption(f"Generated {datetime.now():%A %d %B %Y, %H:%M}")
with top_r:
    if st.button("Sign out"):
        st.session_state["staff_authed"] = False
        st.rerun()

# ── Load sessions ─────────────────────────────────────────────────────────────
if not SESSIONS_FILE.exists():
    st.info("No feedback sessions recorded yet.")
    st.stop()

sessions = []
with SESSIONS_FILE.open() as f:
    for line in f:
        if line.strip():
            sessions.append(json.loads(line))

# Filter to today
today = date.today().isoformat()
today_sessions = [s for s in sessions if s["started"].startswith(today)]

if not today_sessions:
    st.info("No feedback sessions today yet.")
    st.markdown("### All previous sessions")
    today_sessions = sessions  # fall back so the demo still shows something

# ── Top-level metrics ─────────────────────────────────────────────────────────
all_answers = [a for s in today_sessions for a in s["answers"]]
flagged = [a for a in all_answers if a.get("mismatch")]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sessions", len(today_sessions))
m2.metric("Questions answered", len(all_answers))
m3.metric("Mismatch flags", len(flagged))
mismatch_rate = (len(flagged) / max(len(all_answers), 1)) * 100
m4.metric("Flag rate", f"{mismatch_rate:.0f}%")

st.markdown("---")

# ── Distributions ─────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Stated sentiment")
    stated = Counter(a.get("stated_sentiment") or "," for a in all_answers)
    st.bar_chart(pd.DataFrame.from_dict(stated, orient="index", columns=["count"]))

with c2:
    st.markdown("### Dominant facial emotion")
    facial = Counter(a.get("dominant_facial_emotion") or "," for a in all_answers)
    st.bar_chart(pd.DataFrame.from_dict(facial, orient="index", columns=["count"]))

st.markdown("---")

# ── Flagged answers table ─────────────────────────────────────────────────────
st.markdown("### Flagged: polite words, unhappy face")
st.caption("Patients who *said* things were okay but whose face suggested otherwise.")

if flagged:
    rows = []
    for s in today_sessions:
        for a in s["answers"]:
            if a.get("mismatch"):
                rows.append({
                    "Session": s["session_id"],
                    "Question": a["q"],
                    "What they said": a["a"][:120] + ("…" if len(a["a"]) > 120 else ""),
                    "Stated": a["stated_sentiment"],
                    "Face": a["dominant_facial_emotion"],
                    "Negativity": f"{(a.get('facial_negativity') or 0):.0%}",
                })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.success("No mismatches today, patients' words and faces aligned.")

st.markdown("---")

# ── All sessions (expandable) ─────────────────────────────────────────────────
st.markdown("### All sessions today")
for s in today_sessions:
    flag_count = sum(1 for a in s["answers"] if a.get("mismatch"))
    badge = f"🚩 {flag_count} flag(s)" if flag_count else "✓ no flags"
    with st.expander(f"Session {s['session_id']}, {s['started'][11:16]}, {badge}"):
        for a in s["answers"]:
            st.markdown(f"**Q:** {a['q']}")
            st.markdown(f"**A:** {a['a']}")
            cols = st.columns(3)
            cols[0].caption(f"Stated: {a.get('stated_sentiment') or ','}")
            cols[1].caption(f"Face: {a.get('dominant_facial_emotion') or ','}")
            cols[2].caption(
                f"Mismatch: {'YES' if a.get('mismatch') else 'no'}"
            )
            st.markdown("---")
