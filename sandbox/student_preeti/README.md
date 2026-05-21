# EmpathBot — Streamlit UI Scaffold

Calm, accessibility-first Streamlit app for the CS731 emotion-aware healthcare feedback project.

## Run

```bash
pip install -r requirements.txt
streamlit run Home.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Pages

| Page | Path | Purpose |
|---|---|---|
| Home | `Home.py` | Welcome, consent, mode picker |
| Q&A | `pages/1_Ask_Questions.py` | Plain-language answers about AI in healthcare |
| Feedback | `pages/2_Give_Feedback.py` | 4-question post-visit feedback with mismatch detection |
| Staff | `pages/3_Staff_Dashboard.py` | Password-gated daily report (default password: `empath2026`) |

## What's stubbed (and where to plug real components in)

| Stub file | Replace with |
|---|---|
| `utils/bot_stub.py` | `utils/llm_router.py` — your 3-LLM comparison + chosen production model |
| `utils/webcam_stub.py` | `streamlit-webrtc` callback that runs MTCNN + ResNet18 and pushes labels into `st.session_state["emotion_window"]` |
| `utils/mismatch_stub.py` | Same logic, but swap the keyword check for VADER (`vaderSentiment.SentimentIntensityAnalyzer`) |

## Design choices

- **Type ≥ 20pt, headings 26–44pt** — large enough for users with reduced vision
- **Single column, max-width 760px** — no eye-tracking across wide screens
- **Georgia serif body** — higher reading comfort than sans for older users on screen
- **Trust palette**: deep teal `#0f4c5c` primary, warm off-white `#faf7f2` background
- **No emoji decoration** — only used inline where it adds clear meaning (🚩 for staff flags)
- **One action per screen** — no choice overload
- **Mismatch flags are never shown to patients** — ethics: don't accuse people of lying with their face

## Data

Feedback sessions are appended as JSON lines to `data/sessions.jsonl`.
The staff dashboard reads from this file. Delete it to reset.
