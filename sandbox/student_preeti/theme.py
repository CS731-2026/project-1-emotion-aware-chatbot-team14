"""
Central theme / CSS for EmpathBot.

Design principles for the elderly user:
  - Minimum 20pt body text, 28pt+ for headings
  - High contrast (WCAG AAA where possible)
  - Single column, no clutter
  - Trust palette: deep teal primary, warm off-white background
  - Generous spacing
"""
import streamlit as st


_CSS = """
<style>
/* ── Type & base ─────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 20px;
    color: #1a2a33;
    background-color: #faf7f2;
}

/* Streamlit container width — keep narrow for readability */
.main .block-container {
    max-width: 760px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

/* Headings */
h1 { font-size: 44px !important; font-weight: 700; color: #0f4c5c; letter-spacing: -0.5px; }
h2 { font-size: 32px !important; font-weight: 600; color: #0f4c5c; }
h3 { font-size: 26px !important; font-weight: 600; color: #1a2a33; margin-top: 1rem; }
h4 { font-size: 22px !important; font-weight: 600; color: #1a2a33; }

p, li, label, span, div { font-size: 20px; line-height: 1.6; }

/* ── Buttons — large tap targets ──────────────────────────────────────────── */
.stButton > button {
    font-size: 22px !important;
    font-weight: 600 !important;
    padding: 1rem 1.5rem !important;
    border-radius: 12px !important;
    min-height: 64px !important;
    transition: transform 0.1s ease, box-shadow 0.2s ease;
}
.stButton > button:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(15, 76, 92, 0.15);
}
.stButton > button[kind="primary"] {
    background-color: #0f4c5c !important;
    color: #faf7f2 !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover:not(:disabled) {
    background-color: #0a3a47 !important;
}
.stButton > button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea {
    font-size: 20px !important;
    padding: 0.9rem !important;
    border-radius: 10px !important;
    border: 2px solid #d4c9b8 !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #0f4c5c !important;
    box-shadow: 0 0 0 3px rgba(15, 76, 92, 0.15) !important;
}

/* ── Checkboxes — bigger ─────────────────────────────────────────────────── */
.stCheckbox > label { font-size: 20px !important; padding-left: 0.5rem; }
.stCheckbox > label > div[data-baseweb="checkbox"] > div { transform: scale(1.4); }

/* ── Container borders — soft cards ──────────────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff;
    border: 1px solid #e8dfd0 !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    box-shadow: 0 1px 3px rgba(15, 76, 92, 0.04);
}

/* ── Hero block ──────────────────────────────────────────────────────────── */
.eb-hero {
    text-align: center;
    padding: 1rem 0 2rem 0;
}
.eb-hero-kicker {
    font-size: 16px;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #b08968;
    margin-bottom: 1rem;
    font-weight: 600;
}
.eb-hero-title {
    font-size: 52px !important;
    line-height: 1.1;
    margin-bottom: 1.2rem;
}
.eb-hero-sub {
    font-size: 22px;
    color: #4a5a63;
    max-width: 560px;
    margin: 0 auto;
}

.eb-spacer { height: 2rem; }

/* ── Footer ──────────────────────────────────────────────────────────────── */
.eb-footer {
    text-align: center;
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid #e8dfd0;
    color: #7a8a93;
    font-size: 17px;
}

/* ── Chat bubbles for Q&A page ───────────────────────────────────────────── */
.eb-msg-user, .eb-msg-bot {
    padding: 1rem 1.4rem;
    border-radius: 16px;
    margin: 0.6rem 0;
    max-width: 90%;
    font-size: 20px;
    line-height: 1.55;
}
.eb-msg-user {
    background-color: #0f4c5c;
    color: #faf7f2;
    margin-left: auto;
    border-bottom-right-radius: 4px;
}
.eb-msg-bot {
    background-color: #ffffff;
    color: #1a2a33;
    border: 1px solid #e8dfd0;
    border-bottom-left-radius: 4px;
}

/* ── Emotion pill (subtle feedback to staff, hidden from patient) ────────── */
.eb-emotion-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 15px;
    font-weight: 600;
    background-color: #f0eae0;
    color: #4a5a63;
    margin-left: 8px;
}

/* ── Hide Streamlit chrome we don't need ─────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
"""


def apply_theme() -> None:
    """Inject the EmpathBot stylesheet into the current Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)
