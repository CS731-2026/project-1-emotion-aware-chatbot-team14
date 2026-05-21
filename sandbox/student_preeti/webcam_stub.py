"""
Webcam panel stub.

In the real implementation this will:
  1. Use streamlit-webrtc to grab the camera stream
  2. Run MTCNN face crop on each frame
  3. Run ResNet18 emotion inference
  4. Append the label to st.session_state["emotion_window"]

For now it offers a manual emotion selector so the UI and mismatch logic
can be tested end-to-end before the model is wired in.
"""
import streamlit as st


EMOTIONS = ["happy", "neutral", "surprise", "sad", "angry", "disgust", "fear"]


def webcam_panel() -> None:
    """Stub UI for the webcam pane."""
    st.markdown(
        "**Webcam stub** — replace with `streamlit-webrtc` + ResNet18 inferencer.  \n"
        "For testing, set the emotion that the (future) model would currently report:"
    )

    cols = st.columns(len(EMOTIONS))
    for i, emo in enumerate(EMOTIONS):
        with cols[i]:
            if st.button(emo, key=f"emo_btn_{emo}", use_container_width=True):
                # Push 3 copies into the window to simulate a few frames
                st.session_state["emotion_window"].extend([emo] * 3)

    window = st.session_state.get("emotion_window", [])
    st.caption(f"Frames in current window: **{len(window)}**  ·  recent: {window[-15:]}")

    if st.button("Clear window", key="clear_emo"):
        st.session_state["emotion_window"] = []
        st.rerun()
