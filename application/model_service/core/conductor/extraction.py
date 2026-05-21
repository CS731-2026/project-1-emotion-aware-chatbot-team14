"""Extract structured facts at the end of a state.

When the conductor transitions away from a state, the chat router calls
`extract_facts(...)` once with:
  - the state we just left (its `facts_extraction_prompt`)
  - the rendered transcript slice that belongs to that state
  - a reasoning agent capable of running a one-shot JSON-mode call

The result is a dict that goes three places:
  1. session.state_facts[<state name>] — in-memory mirror
  2. an appended line in <profile>.session_facts.jsonl — durable record
  3. a SystemEvent of kind "segment_summary" injected into the
     events buffer — visible to the LLM at the next prompt

Parse failures degrade gracefully: the returned dict carries `_raw`
plus `_error`. Transition never blocks on extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.conductor.state import State
    from core.llm.reasoning_agent import LLMReasoningAgent


def extract_facts(
    agent: "LLMReasoningAgent",
    state: "State",
    segment_slice: list[str],
) -> dict:
    """One-shot extraction call. Returns {} if the state has no prompt."""
    if not state.facts_extraction_prompt:
        return {}
    return agent.extract_json(state.facts_extraction_prompt, segment_slice)
