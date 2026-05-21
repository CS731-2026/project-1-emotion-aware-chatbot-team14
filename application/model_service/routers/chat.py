from typing import Literal, cast
import asyncio
import logging
import random

from fastapi import APIRouter, Request
from pydantic import BaseModel

import time

from core.app_state import HRIAppState
from core.conductor import StateContext
from core.conductor.check_in_spec import CheckInSpec
from core.conductor.extraction import extract_facts
from core.events import SystemEvent
from core.llm.reasoning_agent import Mode, ReasoningResult, Stage
from core.transcript_render import compose_stream
from ws.session import get_session, HarnessSession

logger = logging.getLogger(__name__)
router = APIRouter()

NOISE_SYLLABLES = ["bzzt", "whirr", "tik", "vrrm", "shhh", "klik", "drrt", "ping"]


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


def _run_extraction_on_transition(
    session: HarnessSession,
    prev_state_name: str,
    hri: HRIAppState,
) -> None:
    """Extract facts from the just-ended state and inject a segment_summary.

    Builds the slice from `session.system_events` + `transcript_buffer`
    filtered by t >= session.state_started_at, runs the LLM extraction,
    stores facts in session.state_facts[prev_state_name], and appends a
    SystemEvent of kind="segment_summary" so the next state's LLM call
    sees a {{segment_summary: {id, facts}}} marker at the boundary.

    Failure-tolerant: extract_facts returns {_raw, _error} on a parse
    fail. We still record + emit so the transition completes.
    """
    prev_state = session.conductor.state_named(prev_state_name)
    now = time.time()
    cutoff = session.state_started_at
    slice_events = [e for e in session.system_events if e.t >= cutoff]
    slice_segments = [s for s in session.transcript_buffer if float(s.timestamp) >= cutoff]
    segment_slice = compose_stream(
        slice_segments,
        slice_events,
        confidence_remap=lambda c: c,  # raw conf — extraction doesn't care
    )
    facts: dict = {}
    if prev_state is not None and hri.llm_agent is not None:
        facts = extract_facts(hri.llm_agent, prev_state, segment_slice)
    session.state_facts[prev_state_name] = facts
    session.segment_id_counter += 1
    session.system_events.append(SystemEvent(
        kind="segment_summary",
        t=now,
        payload={"id": session.segment_id_counter, "facts": facts},
    ))
    session.state_started_at = now


async def _step_conductor(
    session: HarnessSession | None,
    *,
    form_completed: bool,
    advance_emission: bool,
    hri: HRIAppState | None = None,
) -> tuple[str | None, str | None, _Surface, CheckInSpec | None, bool]:
    """Advance the per-session conductor by one turn.

    Returns (intention, state_name, surface, spec, transitioned). When there
    is no session yet (chat hit before WS session_start), surface defaults
    to "chat" so the frontend still renders the existing hero.

    When the conductor transitions, runs end-of-state fact extraction on
    the just-left state and emits a segment_summary event into the
    session's events buffer. The extraction is an LLM call that can block
    for several seconds — it runs in a worker thread so the event loop
    keeps servicing WebSocket video frames. `hri` is required for
    extraction; pass None only when the caller knows no transition can
    happen.
    """
    if session is None:
        return None, None, "chat", None, False

    ctx = StateContext(
        turn_in_state=session.turn_in_state,
        form_completed=form_completed,
        advance_emission=advance_emission,
    )
    decision = session.conductor.observe(ctx)        # cheap, on the loop
    if decision.transitioned:
        session.turn_in_state = 0
        if decision.prev_state_name and hri is not None:
            await asyncio.to_thread(
                _run_extraction_on_transition,
                session, decision.prev_state_name, hri,
            )
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

    # Step 0a: walk the conductor with whatever pre-LLM signals we have.
    # The post-LLM advance_emission re-step happens after the reasoner runs.
    intention, state_name, surface, spec, _transitioned = await _step_conductor(
        session, form_completed=body.form_complete, advance_emission=False, hri=hri,
    )
    advance_instruction = (
        session.conductor.current.advance_instruction if session else None
    )

    # When the conductor is sitting in a form state the LLM stays silent —
    # forms drive the surface; chip clicks are structured signals, not
    # things the LLM should react to with prose. A user typing or speaking
    # during a form still lands in transcript_buffer (for future yarn
    # states' context) but doesn't trigger a reply.
    is_form_surface = session is not None and session.conductor.current.kind == "form"

    if is_form_surface:
        result = ReasoningResult(reply="", next_mode=body.mode, next_stage=body.stage)
        debug = _fallback_debug_snapshot(body, "(silent: form state)", intention)
    elif hri.emotion_agent and hri.llm_agent:
        # Step 1: collect the contextual state for this profile and turn.
        history = [{"role": m.role, "content": m.content} for m in body.history]
        transcript_segments = session.transcript_buffer[-20:] if session else []
        system_events = session.system_events[-20:] if session else []

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
            system_events=system_events,
            advance_instruction=advance_instruction,
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
            system_events,
            advance_instruction,
        )
    else:
        latest_emotion = emotion_observations[-1].emotion if emotion_observations else "unknown"
        reply = f"{build_noise_reply()} Latest emotion: {latest_emotion}."
        result = ReasoningResult(reply=reply, next_mode=body.mode, next_stage=body.stage)
        debug = _fallback_debug_snapshot(body, latest_emotion, intention)

    # Step 5: pick up any inline tool emissions from the reply (e.g.
    # [[advance]]) and re-step the conductor so the response carries the
    # post-advance view. The reply text has already had markers stripped.
    advance_emitted = any(e.name == "advance" for e in result.emissions)
    if advance_emitted:
        intention, state_name, surface, spec, _transitioned2 = await _step_conductor(
            session, form_completed=False, advance_emission=True, hri=hri,
        )

    # Attach session-state details to the debug payload so the dashboard
    # panel can render the conductor's view at a glance.
    if debug is not None and session is not None:
        debug["session_state"] = {
            "state_name": state_name,
            "surface": surface,
            "turn_in_state": session.turn_in_state,
            "segment_id": session.segment_id_counter,
            "emissions": [{"name": e.name, "payload": e.payload} for e in result.emissions],
            "state_facts": session.state_facts,
        }

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
