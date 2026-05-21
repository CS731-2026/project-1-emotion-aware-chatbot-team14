"""State machine primitives for the session conductor.

A State represents one logical moment in the session flow — running a
form, an open-chat yarn after a form, a wrap-up exchange, etc. The
conductor walks states forward in order; the LLM sees the state's
`intention_prompt` but never the state's name or any transition
vocabulary.

Later iterations populate the optional fields:
  - `advance_instruction` (i4): the [[advance]] tool-emit instruction
    appended to a yarn state's system prompt
  - `facts_schema_name` + `facts_extraction_prompt` (i5): end-of-state
    fact extraction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from .check_in_spec import CheckInSpec


@dataclass(frozen=True)
class StateContext:
    """Inputs the conductor reads when deciding whether to transition.

    Populated by the chat router from session state + the latest turn's
    parsed signals before calling `Conductor.observe(...)`.
    """

    turn_in_state: int                  # turns since this state began
    form_completed: bool = False        # set by the events stream when a form_complete event lands
    advance_emission: bool = False      # set when the reply carried [[advance]]
    elapsed_in_state: float = 0.0       # seconds since this state began


# A hard-advance rule: pure function of StateContext → should we leave the
# state right now? Keep deterministic; soft conditions live elsewhere.
HardAdvance = Callable[[StateContext], bool]


def _never(_: StateContext) -> bool:
    return False


@dataclass(frozen=True)
class State:
    """One node in the session state machine.

    `name` is internal-only — used by the conductor, JSONL persistence, and
    the debug dashboard. It is never sent to the LLM. The LLM only ever
    sees `intention_prompt` (always) and `advance_instruction` (yarn states,
    later iterations).

    `spec` is the check-in form the frontend mounts when kind == "form".
    Required for form states; ignored for yarn / done.
    """

    name: str
    kind: Literal["form", "yarn", "done"]
    intention_prompt: str = ""
    hard_advance: HardAdvance = field(default=_never)
    spec: Optional[CheckInSpec] = None

    # Filled in by later iterations:
    advance_instruction: Optional[str] = None       # i4
    facts_schema_name: Optional[str] = None         # i5 (label only; schema lives in code)
    facts_extraction_prompt: Optional[str] = None   # i5

    # Late-phase guidance — appended to the intention prompt once the user
    # has been in the state for `late_guidance_after` turns. Used to nudge
    # the LLM toward a soft hand-off (e.g. "ready to give feedback?")
    # without changing the state's first-impression tone.
    late_guidance: Optional[str] = None
    late_guidance_after: int = 0
