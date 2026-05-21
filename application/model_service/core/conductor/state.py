"""State machine primitives for the session conductor.

A State represents one logical moment in the session flow — running a
form, an open-chat yarn after a form, a wrap-up exchange, etc. The
conductor walks states forward in order; the LLM sees the prompt the
state's `tick()` returns but never the state's name or any transition
vocabulary.

States are plain subclassable classes (not frozen dataclasses) so a
state can:
  - keep mutable instance attributes that persist across turns
    (counters, "have we nudged already" flags, summaries built from
    transcript scans)
  - override `tick(ctx)` to deterministically decide what to say and
    whether to hand off, instead of leaning entirely on the LLM to
    reason its way through phases

The base class still covers the common scalar-field case —
`State(...)` with kwargs works for any state whose logic is just
"static prompt, advance on a lambda".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from .check_in_spec import CheckInSpec


@dataclass(frozen=True)
class StateContext:
    """Inputs the conductor reads when calling `state.tick(ctx)`.

    Populated by the chat router from session state + the latest turn's
    parsed signals before each per-turn `Conductor.observe(...)` call.

    Note: `advance_emission` is *not* on this ctx. The LLM's `[[advance]]`
    marker is processed by the conductor's separate
    `handle_emission_advance()` method, not by tick — so a tick can't
    accidentally fire twice per turn.
    """

    turn_in_state: int = 0              # turns the state has been live
    form_completed: bool = False        # set when a form_complete event lands
    elapsed_in_state: float = 0.0       # seconds since this state began


@dataclass(frozen=True)
class TickResult:
    """What `state.tick(ctx)` returns.

    `intention` is the system-prompt text the LLM should see this turn.
    `advance` signals the conductor to hand off to the next state.
    A state can drive a transition purely deterministically by returning
    `advance=True` from tick — no LLM `[[advance]]` emission required.
    """

    intention: str
    advance: bool = False


HardAdvance = Callable[[StateContext], bool]


def _never(_: StateContext) -> bool:
    return False


class State:
    """One node in the session state machine.

    `name` is internal-only — used by the conductor, JSONL persistence,
    and the debug dashboard. It is never sent to the LLM.

    `spec` is the check-in form the frontend mounts when kind == "form".
    Required for form states; ignored for yarn / done.

    The primary extension point is `tick(ctx)`. The default tick
    implements the scalar-field behaviour (intention_prompt +
    optional late_guidance after a turn threshold, advance via
    `hard_advance` lambda). Subclasses override `tick` to:
      - maintain mutable instance attributes across turns
      - phase the prompt deterministically by inspecting ctx and self
      - decide to advance based on the state's own reasoning instead of
        waiting for the LLM to emit `[[advance]]`
    """

    def __init__(
        self,
        *,
        name: str,
        kind: Literal["form", "yarn", "done"],
        intention_prompt: str = "",
        hard_advance: HardAdvance = _never,
        spec: Optional[CheckInSpec] = None,
        advance_instruction: Optional[str] = None,
        facts_schema_name: Optional[str] = None,
        facts_extraction_prompt: Optional[str] = None,
        late_guidance: Optional[str] = None,
        late_guidance_after: int = 0,
    ) -> None:
        self.name = name
        self.kind = kind
        self.intention_prompt = intention_prompt
        self.hard_advance = hard_advance
        self.spec = spec
        self.advance_instruction = advance_instruction
        self.facts_schema_name = facts_schema_name
        self.facts_extraction_prompt = facts_extraction_prompt
        self.late_guidance = late_guidance
        self.late_guidance_after = late_guidance_after

    def tick(self, ctx: StateContext) -> TickResult:
        """Default tick — wraps the scalar-field behaviour.

        Returns the base intention (plus late_guidance once
        `ctx.turn_in_state >= late_guidance_after`) and advances when
        the `hard_advance` lambda fires. LLM `[[advance]]` emissions
        are handled outside tick, in `Conductor.handle_emission_advance`.

        Subclasses with mutable instance state should override and call
        super().tick(ctx) only if they want this default appended to
        their own logic.
        """
        intention = self.intention_prompt
        if self.late_guidance and ctx.turn_in_state >= self.late_guidance_after:
            intention = f"{intention}\n\n{self.late_guidance}".strip()
        return TickResult(intention=intention, advance=self.hard_advance(ctx))
