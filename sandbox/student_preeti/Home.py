"""
EmpathBot — Home / Landing page.

Entry point for the elderly user. Handles:
  1. Friendly welcome
  2. Plain-language consent for webcam use
  3. Choice between Q&A mode and Feedback mode

Run with:  streamlit run Home.py
"""
import streamlit as st
from utils.theme import apply_theme
from utils.session import init_session

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmpathBot — Healthcare Companion",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

apply_theme()
init_session()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="eb-hero">
        <div class="eb-hero-kicker">Your healthcare companion</div>
        <h1 class="eb-hero-title">Hello. I'm EmpathBot.</h1>
        <p class="eb-hero-sub">
            I'm here to help you understand how AI is being used in your care,
            and to listen to how you feel about your visit today.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='eb-spacer'></div>", unsafe_allow_html=True)

# ── Consent ───────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("### Before we begin")
    st.markdown(
        """
        - I may use your **webcam** to see how you're feeling, so I can support you better.
        - I **do not save any video or pictures of you.** Only a label like *"happy"* or *"worried"* is kept.
        - You can stop at any time. Nothing you say will affect your healthcare.
        """
    )
    consent = st.checkbox(
        "I understand and I'm happy to continue.",
        value=st.session_state.get("consent_given", False),
        key="consent_checkbox",
    )
    st.session_state["consent_given"] = consent

st.markdown("<div class='eb-spacer'></div>", unsafe_allow_html=True)

# ── Mode selection ────────────────────────────────────────────────────────────
st.markdown("### What would you like to do?")

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.markdown("#### Ask a question")
        st.markdown(
            "Not sure what the doctor meant by *AI*? "
            "Worried about your data? Ask me anything in plain words."
        )
        if st.button(
            "Start asking questions",
            type="primary",
            use_container_width=True,
            disabled=not consent,
            key="btn_ask",
        ):
            st.switch_page("pages/1_Ask_Questions.py")

with col2:
    with st.container(border=True):
        st.markdown("#### Give feedback")
        st.markdown(
            "Tell me how your visit went today. "
            "Just a few short questions — there are no wrong answers."
        )
        if st.button(
            "Start giving feedback",
            type="primary",
            use_container_width=True,
            disabled=not consent,
            key="btn_feedback",
        ):
            st.switch_page("pages/2_Give_Feedback.py")

if not consent:
    st.info("Please tick the box above to continue.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="eb-footer">
        Need help? Ask a member of staff at the reception desk.
    </div>
    """,
    unsafe_allow_html=True,
)
