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
from core.llm.reasoning_agent import ReasoningResult
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
        "intention": intention,
        "system_prompt": None,
        "history_window": 0,
        "history_messages": body.history,
        "emotional_context": f"Fallback mode. Latest emotion: {latest_emotion}.",
        "transcript_lines": [],
        "prompt_messages": [],
    }


_Surface = Literal["chat", "checkin", "done"]


def generate_yarn_opener(
    session: HarnessSession,
    hri: HRIAppState,
    intention: str,
) -> str | None:
    """Run the LLM once with no user message to produce a yarn-opening reply.

    Called right after the conductor transitions into a yarn state via a
    form_complete event — there's no user chat turn to ride; we want the
    assistant to acknowledge the form answers and open the next phase.

    `intention` is the conductor's tick output for the new state's first
    turn — passed in by the caller so subclass tick() logic shapes the
    opener.

    Returns the cleaned reply text, or None if the LLM agent isn't loaded
    or the call fails.
    """
    if hri.llm_agent is None or hri.emotion_agent is None:
        return None
    transcript_segments = session.transcript_buffer[-20:]
    system_events = session.system_events[-20:]
    dominant = (
        max(session.emotion_counter, key=session.emotion_counter.get)
        if session.emotion_counter else None
    )
    emotional_context = hri.emotion_agent.analyse(
        session.emotion_buffer.history(), transcript_segments, dominant,
    )
    current = session.conductor.current
    try:
        result = hri.llm_agent.reason(
            "",  # no user input — the LLM is opening the yarn from the
                 # intention prompt + recent form_answer events
            emotional_context,
            [],
            transcript_segments,
            intention,
            system_events,
            current.advance_instruction,
        )
    except Exception:
        logger.exception("yarn opener LLM call failed")
        return None
    return result.reply.strip() or None


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
    # Strip the (conf NN%) tag from speech lines — fact extraction doesn't
    # benefit from per-line confidence, and rendering it with a different
    # remap than the main conversation would just create inconsistency.
    segment_slice = compose_stream(
        slice_segments,
        slice_events,
        confidence_remap=lambda _c: None,
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
    hri: HRIAppState | None = None,
) -> tuple[str | None, str | None, _Surface, CheckInSpec | None, bool]:
    """Per-turn observe call. Ticks the current state exactly once.

    Returns (intention, state_name, surface, spec, transitioned). When
    there is no session yet (chat hit before WS session_start), surface
    defaults to "chat" so the frontend still renders the existing hero.

    On transition, runs end-of-state fact extraction in a worker thread
    so the LLM call doesn't block the WS event loop. `hri` is required
    for that; pass None only when the caller knows no transition can
    happen.

    `[[advance]]` emissions arrive AFTER the LLM call and are handled by
    `_handle_advance_emission` below — they don't tick.
    """
    if session is None:
        return None, None, "chat", None, False

    ctx = StateContext(
        turn_in_state=session.turn_in_state,
        form_completed=form_completed,
        elapsed_in_state=max(0.0, time.time() - session.state_started_at),
    )
    decision = session.conductor.observe(ctx)
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


async def _handle_advance_emission(
    session: HarnessSession | None,
    *,
    hri: HRIAppState | None = None,
) -> tuple[str | None, str | None, _Surface, CheckInSpec | None, bool]:
    """Post-LLM: the reply carried `[[advance]]`. Walk forward without
    ticking the previous state again (its tick already ran this turn).
    Runs end-of-state extraction on transition. Mirrors _step_conductor's
    return shape for the chat router caller.
    """
    if session is None:
        return None, None, "chat", None, False
    decision = session.conductor.handle_emission_advance()
    if decision.transitioned:
        session.turn_in_state = 0
        if decision.prev_state_name and hri is not None:
            await asyncio.to_thread(
                _run_extraction_on_transition,
                session, decision.prev_state_name, hri,
            )
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
    # form_complete arrives over the WebSocket, never via /chat — so this
    # pre-LLM observe just ticks the current state with form_completed=False.
    intention, state_name, surface, spec, _transitioned = await _step_conductor(
        session, form_completed=False, hri=hri,
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
        result = ReasoningResult(reply="")
        debug = _fallback_debug_snapshot(body, "(silent: form state)", intention)
    elif hri.emotion_agent and hri.llm_agent:
        # Step 1: collect the contextual state for this profile and turn.
        history = [{"role": m.role, "content": m.content} for m in body.history]
        transcript_segments = session.transcript_buffer[-20:] if session else []
        system_events = session.system_events[-20:] if session else []

        # Step 2: summarise the emotional signal into a compact reasoning input.
        session_dominant = (
            max(session.emotion_counter, key=session.emotion_counter.get)
            if session and session.emotion_counter else None
        )
        emotional_context = hri.emotion_agent.analyse(
            emotion_observations, transcript_segments, session_dominant
        )

        # Step 3: expose the current reasoning snapshot for the debug UI.
        debug = hri.llm_agent.debug_snapshot(
            body.message,
            emotional_context,
            history,
            transcript_segments,
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
            intention,
            system_events,
            advance_instruction,
        )
    else:
        latest_emotion = emotion_observations[-1].emotion if emotion_observations else "unknown"
        reply = f"{build_noise_reply()} Latest emotion: {latest_emotion}."
        result = ReasoningResult(reply=reply)
        debug = _fallback_debug_snapshot(body, latest_emotion, intention)

    # Step 5: pick up any inline tool emissions from the reply (e.g.
    # [[advance]]) and ask the conductor to process the transition WITHOUT
    # re-ticking the previous state — that would double-count the turn.
    # The reply text has already had markers stripped.
    advance_emitted = any(e.name == "advance" for e in result.emissions)
    if advance_emitted:
        intention, state_name, surface, spec, _transitioned2 = await _handle_advance_emission(
            session, hri=hri,
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
        debug=debug,
    )
