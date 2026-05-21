"""Inline tool emissions parsed out of the LLM's reply text.

The LLM is occasionally instructed (via a state's `advance_instruction`)
to append a marker like `[[advance]]` to its reply when a particular
condition is met. The reasoning agent strips the marker before the user
sees the reply, and emits a ToolEmission the conductor can read.

Today there's exactly one emission: `advance` (parameterless). The
dataclass + parser are deliberately set up to extend to:
  - markers with payloads: `[[record_fact: {"key": "..."}]]`
  - additional kinds: `[[suggest_check_in: feedback]]`,
    `[[note: "..."]]`, etc.
without changing the conductor's interface — it always reads a list of
ToolEmission(name, payload) values.

When we eventually move to provider-native tool calling, the parser
implementation changes (reads `tool_calls` off the message), but the
returned list shape stays identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolEmission:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)


# Today we only recognise [[advance]] — case-insensitive, optionally
# surrounded by whitespace, last thing in the reply (own line preferred).
# Extending to parametric markers is one more capture group + JSON parse.
_ADVANCE_RE = re.compile(r"\n?\s*\[\[\s*advance\s*\]\]\s*$", re.IGNORECASE)


def extract_emissions(text: str) -> tuple[str, list[ToolEmission]]:
    """Pull recognised inline markers out of `text`.

    Returns the cleaned text (markers stripped) and the list of emissions
    found. Robust to trailing whitespace / case / leading newline.
    """
    emissions: list[ToolEmission] = []
    cleaned = text

    new = _ADVANCE_RE.sub("", cleaned)
    if new != cleaned:
        emissions.append(ToolEmission(name="advance"))
        cleaned = new

    return cleaned.rstrip(), emissions
