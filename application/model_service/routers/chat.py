from typing import Literal, cast
import asyncio
import logging
import random

from fastapi import APIRouter, Request
from pydantic import BaseModel

from core.app_state import HRIAppState
from core.conductor import StateContext
from core.conductor.check_in_spec import CheckInSpec
from core.llm.reasoning_agent import Mode, ReasoningResult, Stage
from ws.session import get_session, HarnessSession

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
    # Set by the frontend when the user has just answered the last question
    # of an active check-in form. Drives the conductor's qa_form → next-state
    # transition. Iteration 3 will replace this with a typed event on the
    # WebSocket so chip clicks stop routing through /chat at all.
    form_complete: bool = False
    # Legacy fields; ignored by the conductor path. Removed in iteration 7.
    mode: Mode = "qa"
    stage: Stage | None = None


class ChatView(BaseModel):
    """What the frontend should render. Derived from the conductor's current state."""

    surface: Literal["chat", "checkin", "done"]
    spec: CheckInSpec | None = None
    # Internal debug-only fields. Never used by the LLM; surfaced for the
    # debug dashboard.
    intention: str | None = None
    state_name: str | None = None


class ChatResponse(BaseModel):
    response: str
    view: ChatView
    # Legacy next_mode / next_stage fields stay on the response while the
    # frontend is migrating; the frontend now reads `view` instead. Removed
    # in iteration 7.
    next_mode: Mode = "qa"
    next_stage: Stage | None = None
    debug: dict | None = None


def build_noise_reply() -> str:
    """Return a random noise string used when LLM agents are not loaded."""
    burst = " ".join(random.choice(NOISE_SYLLABLES) for _ in range(random.randint(6, 10)))
    return f"Test harness reply: {burst}."


def _fallback_debug_snapshot(body: ChatRequest, latest_emotion: str, intention: str | None) -> dict:
    """Return a debug payload when the real reasoning pipeline is unavailable."""
    return {
        "provider": None,
        "model": None,
        "current_message": body.message,
        "mode": body.mode,
        "stage": body.stage,
        "intention": intention,
        "system_prompt": None,
        "history_window": 0,
        "history_messages": body.history,
        "emotional_context": f"Fallback mode. Latest emotion: {latest_emotion}.",
        "transcript_lines": [],
        "prompt_messages": [],
    }


_Surface = Literal["chat", "checkin", "done"]


def _step_conductor(
    session: HarnessSession | None,
    form_completed: bool,
) -> tuple[str | None, str | None, _Surface, CheckInSpec | None, bool]:
    """Advance the per-session conductor by one turn.

    Returns (intention, state_name, surface, spec, transitioned). When there
    is no session yet (chat hit before WS session_start), surface defaults to
    "chat" so the frontend still renders the existing hero.
    """
    if session is None:
        return None, None, "chat", None, False

    ctx = StateContext(
        turn_in_state=session.turn_in_state,
        form_completed=form_completed,
        advance_emission=False,      # i4 plumbs this in
    )
    decision = session.conductor.observe(ctx)
    if decision.transitioned:
        session.turn_in_state = 0
    else:
        session.turn_in_state += 1
    return (
        decision.intention,
        decision.state.name,
        decision.surface,
        decision.state.spec,
        decision.transitioned,
    )


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

    # Step 0: walk the session conductor forward by one turn. Produces the
    # intention the reasoner will use and the view the frontend will render.
    intention, state_name, surface, spec, _transitioned = _step_conductor(
        session, form_completed=body.form_complete,
    )

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
            intention=intention,
        )

        # Step 4: run the current LLM reasoning pipeline to produce a structured result.
        # Offload to a worker thread — providers like ClaudeCodeProvider spawn a
        # subprocess and block; running them on the event loop stalls the WS
        # face-detection pipeline until the LLM returns.
        result = await asyncio.to_thread(
            hri.llm_agent.reason,
            body.message,
            emotional_context,
            history,
            transcript_segments,
            body.mode,
            body.stage,
            intention,
        )
    else:
        latest_emotion = emotion_observations[-1].emotion if emotion_observations else "unknown"
        reply = f"{build_noise_reply()} Latest emotion: {latest_emotion}."
        # Stub mode never transitions — keep the caller's state.
        result = ReasoningResult(reply=reply, next_mode=body.mode, next_stage=body.stage)
        debug = _fallback_debug_snapshot(body, latest_emotion, intention)

    # Soft turn cap: count this user message plus prior user turns in history.
    user_turns = sum(1 for m in body.history if m.role == "user") + 1
    if user_turns >= TURN_CAP and result.next_mode != "done":
        logger.info("turn cap reached (%d) — forcing next_mode='done'", user_turns)
        result = ReasoningResult(reply=result.reply, next_mode="done", next_stage=None)

    return ChatResponse(
        response=result.reply,
        view=ChatView(
            surface=surface,
            spec=spec if surface == "checkin" else None,
            intention=intention,
            state_name=state_name,
        ),
        next_mode=result.next_mode,
        next_stage=result.next_stage,
        debug=debug,
    )
