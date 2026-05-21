"""State machine primitives for the session conductor.

A State represents one logical moment in the session flow — running a
form, an open-chat yarn after a form, a wrap-up exchange, etc. The
conductor walks states forward in order; the LLM sees the prompt
produced by `state.intention_for(ctx)` but never the state's name or
any transition vocabulary.

States are a subclassable plain class (not a frozen dataclass) so a
state can carry instance-level mutable bookkeeping (counters, "have
we nudged already" flags, etc.) and override per-turn logic. The base
class still covers the common scalar-field case — instantiating
`State(...)` with kwargs works for any state that doesn't need
custom behaviour beyond `late_guidance` + a `hard_advance` lambda.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from .check_in_spec import CheckInSpec


@dataclass(frozen=True)
class StateContext:
    """Inputs the conductor reads when deciding whether to transition.

    Populated by the chat router from session state + the latest turn's
    parsed signals before calling `Conductor.observe(...)`.
    """

    turn_in_state: int = 0              # turns since this state began
    form_completed: bool = False        # set by the events stream when a form_complete event lands
    advance_emission: bool = False      # set when the reply carried [[advance]]
    elapsed_in_state: float = 0.0       # seconds since this state began


# A hard-advance rule: pure function of StateContext → should we leave the
# state right now? Keep deterministic; soft conditions live elsewhere.
HardAdvance = Callable[[StateContext], bool]


def _never(_: StateContext) -> bool:
    return False


class State:
    """One node in the session state machine.

    `name` is internal-only — used by the conductor, JSONL persistence,
    and the debug dashboard. It is never sent to the LLM. The LLM only
    ever sees `intention_for(ctx)` output and (for yarns)
    `advance_instruction`.

    `spec` is the check-in form the frontend mounts when kind == "form".
    Required for form states; ignored for yarn / done.

    Subclasses can override:
      - `intention_for(ctx)` — build the system prompt for this turn,
        e.g. branching on `ctx.turn_in_state` or on instance attributes
        the subclass maintains across turns.
      - `should_advance(ctx)` — decide whether to hand off to the next
        state, replacing the default (hard_advance lambda + [[advance]]
        emission for yarn states).
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

    def intention_for(self, ctx: StateContext) -> str:
        """Build the prompt text the LLM should see this turn.

        Default behaviour: return `intention_prompt`, optionally with
        `late_guidance` appended once `ctx.turn_in_state` reaches
        `late_guidance_after`. Subclasses can override for arbitrary
        phase or counter-based logic.
        """
        text = self.intention_prompt
        if self.late_guidance and ctx.turn_in_state >= self.late_guidance_after:
            text = f"{text}\n\n{self.late_guidance}".strip()
        return text

    def should_advance(self, ctx: StateContext) -> bool:
        """Should the conductor hand off to the next state this turn?

        Default behaviour: the `hard_advance` rule fires (form
        completion, turn / time cap), or this is a yarn that exposes
        an `advance_instruction` and the LLM emitted `[[advance]]`.
        """
        if self.hard_advance(ctx):
            return True
        return (
            self.kind == "yarn"
            and self.advance_instruction is not None
            and ctx.advance_emission
        )
