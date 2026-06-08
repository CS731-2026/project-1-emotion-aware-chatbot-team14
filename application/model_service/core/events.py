"""Typed system events that flow through the merged transcript stream.

The LLM sees the conversation as a chronological mix of two kinds of
entries:
  - user speech (transcript_buffer entries)
  - system events (this module)

Speech is rendered as plain text. System events render as
`[t.t] {{kind: payload_json}}` so the LLM can recognise them as
observations from other systems, not things the user said.

System events also give non-LLM consumers, staff dashboard,
analytics, replay tools, a structured log they can `grep` over.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


EventKind = Literal[
    "form_answer",       # user picked a chip in a check-in form
    "form_complete",     # all questions in a form have been answered
    "emotion_window",    # sustained emotion signal from the face detector
    "segment_summary",   # facts extracted at the end of a state (i5)
    "silence",           # detected pause of significant duration
]


@dataclass(frozen=True)
class SystemEvent:
    kind: EventKind
    t: float                              # session-clock seconds
    payload: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Render to a single transcript line in the {{…}} convention."""
        return f"[{self.t:.1f}s] {{{{{self.kind}: {json.dumps(self.payload, separators=(',', ':'))}}}}}"
