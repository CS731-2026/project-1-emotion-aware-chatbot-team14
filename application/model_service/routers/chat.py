from typing import cast
import random

from fastapi import APIRouter, Request
from pydantic import BaseModel

from core.app_state import HRIAppState
from core.llm.base import Message
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


def build_noise_reply() -> str:
    """Return a random noise string used when LLM agents are not loaded."""
    burst = " ".join(random.choice(NOISE_SYLLABLES) for _ in range(random.randint(6, 10)))
    return f"Test harness reply: {burst}."


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """POST /api/v1/chat — generate an emotion-aware LLM reply.

    Reads the active WebSocket session's emotion buffer for the given profile,
    passes it through EmotionalReasoningAgent to produce a context string, then
    feeds that context + message history into LLMReasoningAgent. Falls back to
    a noise reply if agents are not loaded (dev/stub mode).
    """
    hri = cast(HRIAppState, request.app.state.hri)
    session = get_session(body.profile_id)
    emotion_observations = session.emotion_buffer.history() if session else []

    if hri.emotion_agent and hri.llm_agent:
        history: list[Message] = [
            cast(Message, {"role": m.role, "content": m.content})
            for m in body.history
        ]
        transcript_segments = session.transcript_buffer[-20:] if session else []

        # Emotional context: face-based signal → EmotionalReasoningAgent
        emotional_context = hri.emotion_agent.analyse(emotion_observations, transcript_segments)

        # Both inputs (emotional_context + transcript_segments) feed LLM Reasoning
        # separately, as per the architecture spec.
        response = hri.llm_agent.reason(body.message, emotional_context, history, transcript_segments)
    else:
        latest_emotion = emotion_observations[-1].emotion if emotion_observations else "unknown"
        response = f"{build_noise_reply()} Latest emotion: {latest_emotion}."

    return ChatResponse(response=response)
