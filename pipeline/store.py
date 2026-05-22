"""Phase-to-phase handoff bag.

Each phase puts the objects it produces under a key from
`pipeline.keys`; later phases get them back, type-asserted at the
boundary so a misnamed key or wrong-type clobber fails loudly instead
of propagating None.

Strict by default: `put` refuses to overwrite an existing key — use
`replace` when overwriting is the intent (e.g. a re-eval phase
replacing an earlier `EVAL_REPORT`). Catches the easy bug where two
phases accidentally share a key.
"""

from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


class Store:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def put(self, key: str, value: Any) -> None:
        """Insert a value under `key`. Raises if the key is already set."""
        if key in self._data:
            raise KeyError(
                f"store already has {key!r} "
                f"(holding a {type(self._data[key]).__name__}). "
                f"Use replace() if overwriting is intentional."
            )
        self._data[key] = value

    def replace(self, key: str, value: Any) -> None:
        """Overwrite-allowed sibling of put(). Use when a phase legitimately
        produces a new version of an existing key (e.g. re-evaluation)."""
        self._data[key] = value

    def get(self, key: str, expected: type[T]) -> T:
        """Fetch a value by key, asserting it's an instance of `expected`.

        Both missing-key and wrong-type cases raise with a message naming
        the available keys so the caller can spot a typo or a skipped
        earlier phase without reading the trace.
        """
        if key not in self._data:
            raise KeyError(
                f"store has no entry for {key!r}. "
                f"available: {sorted(self._data.keys())} — "
                f"did an earlier phase fail or get skipped?"
            )
        value = self._data[key]
        if not isinstance(value, expected):
            raise TypeError(
                f"store[{key!r}] is {type(value).__name__}, "
                f"expected {expected.__name__}"
            )
        return value

    def has(self, key: str) -> bool:
        return key in self._data
