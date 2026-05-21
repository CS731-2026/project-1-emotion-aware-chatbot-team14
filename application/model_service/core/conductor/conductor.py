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

        Walks one step:
          1. Ask the current state via `should_advance(ctx)`.
          2. If it returns True (and we're not already on the last
             state), move forward.
          3. Build the new current state's intention via
             `intention_for(ctx)` — using a fresh ctx with turn 0 on
             transitions so the new state sees its first turn.
        """
        prev = self.current
        transitioned = False
        if self._idx < len(self._states) - 1 and self.current.should_advance(ctx):
            self._idx += 1
            transitioned = True
        cur = self.current
        intention_ctx = StateContext() if transitioned else ctx
        return ConductorDecision(
            state=cur,
            intention=cur.intention_for(intention_ctx),
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
