"""
Stub bot replies. Replace with `utils.llm_router.get_reply` once your
3-LLM comparison module is implemented.

Keeps the UI runnable end-to-end without API keys.
"""
import random


_CANNED = {
    "scan": (
        "When the doctor uses AI to look at your scan, it's like a second pair "
        "of eyes that helps spot things faster. A real doctor still looks at "
        "every scan and makes the final decision — the AI doesn't decide on its own."
    ),
    "data": (
        "Your health information is kept private. The AI only looks at what it "
        "needs to do its job, and the clinic follows strict rules about who can "
        "see your records. You can ask your GP exactly what is stored, any time."
    ),
    "doctor": (
        "Yes — a real doctor reviews everything. The AI is a tool that supports "
        "the doctor, like a calculator supports an accountant. It doesn't replace them."
    ),
    "decide": (
        "The computer doesn't make the decision. Your GP does. The AI just helps "
        "by sorting through information so your doctor has more time to talk with you."
    ),
}


def mock_bot_reply(user_text: str, emotion: str | None = None) -> str:
    """Return a canned, plain-language reply. Soften tone if user looks worried."""
    text = user_text.lower()
    reply = None
    for keyword, response in _CANNED.items():
        if keyword in text:
            reply = response
            break
    if reply is None:
        reply = (
            "That's a really good question. The short answer is that AI is being "
            "used as a helper for the doctors here, never as a replacement. "
            "Would you like me to explain a bit more, or ask the receptionist "
            "to come over?"
        )

    # If the webcam picked up sustained worry/sadness, add a gentle reassurance
    if emotion in {"sad", "fear", "angry"}:
        reply = (
            "I can see this might feel a bit worrying — that's completely okay. "
            + reply
        )
    return reply
