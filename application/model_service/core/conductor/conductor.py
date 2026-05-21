"""Session conductor — per-session state machine walker.

Holds a list of States and a pointer to the current one. Each user turn
the chat router calls `observe(ctx)` once; the conductor invokes
`state.tick(ctx)` exactly once on the current state and may transition.
After the LLM has replied, if the reply carried `[[advance]]`, the
router calls `handle_emission_advance()` — a separate entry that
transitions without re-ticking, so per-turn counters stay correct.

The conductor returns a `ConductorDecision` carrying:
  - the (possibly new) current state
  - its tick'd intention prompt (for the reasoning agent to inject)
  - the rendering surface ("chat" | "checkin" | "done") for the frontend
  - a `transitioned` flag and the just-ended state's name when applicable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .state import State, StateContext


Surface = Literal["chat", "checkin", "done"]


@dataclass(frozen=True)
class ConductorDecision:
    state: State
    intention: str
    surface: Surface
    transitioned: bool
    prev_state_name: Optional[str] = None


class Conductor:
    """Walks a fixed list of States forward as conditions fire."""

    def __init__(self, states: list[State]) -> None:
        if not states:
            raise ValueError("Conductor needs at least one state")
        self._states = states
        self._idx = 0

    @property
    def current(self) -> State:
        return self._states[self._idx]

    def state_named(self, name: str) -> State | None:
        """Look up a state by its internal name. None if not in this flow."""
        return next((s for s in self._states if s.name == name), None)

    def observe(self, ctx: StateContext) -> ConductorDecision:
        """Per-turn entry. Tick the current state exactly once.

        If the tick result says `advance` and we're not already on the
        last state, transition forward and tick the new state with a
        fresh ctx (turn_in_state=0). The intention returned is whichever
        tick was last evaluated — either the current state's (no
        transition) or the new state's (transition).
        """
        cur = self.current
        result = cur.tick(ctx)
        if result.advance and self._idx < len(self._states) - 1:
            prev = cur
            self._idx += 1
            new_state = self.current
            new_result = new_state.tick(StateContext())
            return ConductorDecision(
                state=new_state,
                intention=new_result.intention,
                surface=_surface_for(new_state),
                transitioned=True,
                prev_state_name=prev.name,
            )
        return ConductorDecision(
            state=cur,
            intention=result.intention,
            surface=_surface_for(cur),
            transitioned=False,
            prev_state_name=None,
        )

    def handle_emission_advance(self) -> ConductorDecision:
        """Post-LLM entry. Process an `[[advance]]` emission.

        Only fires a transition when the current state is a yarn that
        exposes an `advance_instruction` (i.e. the LLM was authorised
        to emit the marker). Does NOT call `tick()` on the current
        state — that already happened pre-LLM. The new state's `tick`
        runs once with a fresh ctx.

        If no transition is possible (current is the last state, or
        not a yarn with an advance_instruction), returns the current
        state's view unchanged.
        """
        cur = self.current
        can_advance = (
            cur.kind == "yarn"
            and cur.advance_instruction is not None
            and self._idx < len(self._states) - 1
        )
        if not can_advance:
            return ConductorDecision(
                state=cur,
                intention=cur.intention_prompt,
                surface=_surface_for(cur),
                transitioned=False,
                prev_state_name=None,
            )
        prev = cur
        self._idx += 1
        new_state = self.current
        new_result = new_state.tick(StateContext())
        return ConductorDecision(
            state=new_state,
            intention=new_result.intention,
            surface=_surface_for(new_state),
            transitioned=True,
            prev_state_name=prev.name,
        )


def _surface_for(state: State) -> Surface:
    if state.kind == "form":
        return "checkin"
    if state.kind == "done":
        return "done"
    return "chat"
