from typing import cast
import logging
import random

from fastapi import APIRouter, Request
from pydantic import BaseModel

from core.app_state import HRIAppState
from core.llm.reasoning_agent import Mode, ReasoningResult, Stage
from ws.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter()

NOISE_SYLLABLES = ["bzzt", "whirr", "tik", "vrrm", "shhh", "klik", "drrt", "ping"]

# Soft safety cap. If the reasoner never emits next_mode="done", force it after
# this many user turns so a misbehaving model can't trap a study participant.
TURN_CAP = 30


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    profile_id: str
    message: str
    history: list[ChatMessage] = []
    # Which page / conversational mode the user is in. Defaults preserve today's behaviour.
    mode: Mode = "qa"
    # Within `qa` mode, the current scripted stage. None means stage-agnostic.
    stage: Stage | None = None


class ChatResponse(BaseModel):
    response: str
    # Where the reasoner thinks the conversation should go next. Frontend mirrors
    # these into its conversation state to drive the view that renders.
    next_mode: Mode = "qa"
    next_stage: Stage | None = None
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
        "mode": body.mode,
        "stage": body.stage,
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
    hri = cast(HRIAppState, request.app.state.hri)
    session = get_session(body.profile_id)
    emotion_observations = session.emotion_buffer.history() if session else []

    if hri.emotion_agent and hri.llm_agent:
        # Step 1: collect the contextual state for this profile and turn.
        history = [{"role": m.role, "content": m.content} for m in body.history]
        transcript_segments = session.transcript_buffer[-20:] if session else []

        # Step 2: summarise the emotional signal into a compact reasoning input.
        emotional_context = hri.emotion_agent.analyse(emotion_observations, transcript_segments)

        # Step 3: expose the current reasoning snapshot for the debug UI.
        debug = hri.llm_agent.debug_snapshot(
            body.message,
            emotional_context,
            history,
            transcript_segments,
            mode=body.mode,
            stage=body.stage,
        )

        # Step 4: run the current LLM reasoning pipeline to produce a structured result.
        result = hri.llm_agent.reason(
            body.message,
            emotional_context,
            history,
            transcript_segments,
            mode=body.mode,
            stage=body.stage,
        )
    else:
        latest_emotion = emotion_observations[-1].emotion if emotion_observations else "unknown"
        reply = f"{build_noise_reply()} Latest emotion: {latest_emotion}."
        # Stub mode never transitions — keep the caller's state.
        result = ReasoningResult(reply=reply, next_mode=body.mode, next_stage=body.stage)
        debug = _fallback_debug_snapshot(body, latest_emotion)

    # Soft turn cap: count this user message plus prior user turns in history.
    user_turns = sum(1 for m in body.history if m.role == "user") + 1
    if user_turns >= TURN_CAP and result.next_mode != "done":
        logger.info("turn cap reached (%d) — forcing next_mode='done'", user_turns)
        result = ReasoningResult(reply=result.reply, next_mode="done", next_stage=None)

    return ChatResponse(
        response=result.reply,
        next_mode=result.next_mode,
        next_stage=result.next_stage,
        debug=debug,
    )
