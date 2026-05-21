"""Session conductor — per-session state machine walker.

Holds a list of States and a pointer to the current one. Each turn the
chat router calls `observe(ctx)` with the latest StateContext; the
conductor evaluates the current state's `hard_advance` rule and, if it
fires, walks to the next state. Forward-only; never backtracks.

The conductor returns a `ConductorDecision` carrying:
  - the (possibly new) current state
  - its intention prompt (for the reasoning agent to inject)
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
        """Possibly advance to the next state and return the resulting view.

        Order of evaluation:
          1. hard_advance(ctx) — deterministic rule (form completion, turn cap).
          2. soft advance: yarn states with ctx.advance_emission=True
             advance when the LLM has appended an [[advance]] marker.

        hard wins if both fire on the same turn — same end state, same
        net effect.
        """
        prev = self.current
        transitioned = False
        if self._idx < len(self._states) - 1:
            cur = self.current
            should_advance = cur.hard_advance(ctx) or (
                cur.kind == "yarn"
                and cur.advance_instruction is not None
                and ctx.advance_emission
            )
            if should_advance:
                self._idx += 1
                transitioned = True
        cur = self.current
        return ConductorDecision(
            state=cur,
            intention=cur.intention_prompt,
            surface=_surface_for(cur),
            transitioned=transitioned,
            prev_state_name=prev.name if transitioned else None,
        )


def _surface_for(state: State) -> Surface:
    if state.kind == "form":
        return "checkin"
    if state.kind == "done":
        return "done"
    return "chat"
