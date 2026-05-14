from fastapi import APIRouter, Request
from pydantic import BaseModel
import random

from ws.session import get_session

router = APIRouter()

NOISE_SYLLABLES = ["bzzt", "whirr", "tik", "vrrm", "shhh", "klik", "drrt", "ping"]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    profile_id: str
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str
    debug: dict | None = None


def build_noise_reply() -> str:
    """Return a random noise string used when LLM agents are not loaded."""
    burst = " ".join(random.choice(NOISE_SYLLABLES) for _ in range(random.randint(6, 10)))
    return f"Test harness reply: {burst}."


def _fallback_debug_snapshot(body: ChatRequest, latest_emotion: str) -> dict:
    """Return a debug payload when the real reasoning pipeline is unavailable."""
    return {
        "provider": None,
        "model": None,
        "current_message": body.message,
        "system_prompt": None,
        "history_window": 0,
        "history_messages": body.history,
        "emotional_context": f"Fallback mode. Latest emotion: {latest_emotion}.",
        "transcript_lines": [],
        "prompt_messages": [],
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """POST /api/v1/chat — generate an emotion-aware LLM reply.

    Reads the active WebSocket session's emotion buffer for the given profile,
    passes it through EmotionalReasoningAgent to produce a context string, then
    feeds that context + message history into LLMReasoningAgent. Falls back to
    a noise reply if agents are not loaded (dev/stub mode).
    """
    session = get_session(body.profile_id)
    emotion_observations = session.emotion_buffer.history() if session else []

    emotion_agent = getattr(request.app.state, "emotion_agent", None)
    llm_agent = getattr(request.app.state, "llm_agent", None)

    if emotion_agent and llm_agent:
        # Step 1: collect the contextual state for this profile and turn.
        history = [{"role": m.role, "content": m.content} for m in body.history]
        transcript_segments = session.transcript_buffer[-20:] if session else []

        # Step 2: summarise the emotional signal into a compact reasoning input.
        emotional_context = emotion_agent.analyse(emotion_observations, transcript_segments)

        # Step 3: expose the current reasoning snapshot for the debug UI.
        debug = llm_agent.debug_snapshot(
            body.message,
            emotional_context,
            history,
            transcript_segments,
        )

        # Step 4: run the current LLM reasoning pipeline to produce a reply.
        response = llm_agent.reason(body.message, emotional_context, history, transcript_segments)
    else:
        latest_emotion = emotion_observations[-1].emotion if emotion_observations else "unknown"
        response = f'{build_noise_reply()} Latest emotion: {latest_emotion}.'
        debug = _fallback_debug_snapshot(body, latest_emotion)

    return ChatResponse(response=response, debug=debug)
