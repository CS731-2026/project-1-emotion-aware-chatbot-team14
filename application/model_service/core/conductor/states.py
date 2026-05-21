"""State definitions for the session flow.

iteration 1 ships a single open-chat state — behaves identically to the
pre-conductor app. iteration 2 onwards replaces this list with the real
forward flow (qa_form → post_qa_yarn → feedback_form → post_feedback_yarn
→ closing_yarn → done).
"""

from .state import State

OPEN_CHAT = State(
    name="open_chat",
    kind="yarn",
    intention_prompt=(
        "You are guiding a calm, voice-first conversation. "
        "Stay focused on listening and helping the user feel heard. "
        "Speak naturally, warmly, clearly, and briefly."
    ),
)

SESSION_FLOW: list[State] = [OPEN_CHAT]
