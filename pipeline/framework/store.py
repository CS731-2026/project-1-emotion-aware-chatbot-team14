"""Phase-to-phase handoff bag. Strict put (no clobber), typed get."""

from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


class Store:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def put(self, key: str, value: Any) -> None:
        # Refuses overwrite; use replace() if that's the intent. Catches
        # two phases accidentally sharing a key.
        if key in self._data:
            raise KeyError(
                f"store already has {key!r} "
                f"(holding {type(self._data[key]).__name__}). "
                f"Use replace() if overwriting is intentional."
            )
        self._data[key] = value

    def replace(self, key: str, value: Any) -> None:
        # For phases that legitimately produce a new version (e.g. re-eval).
        self._data[key] = value

    def get(self, key: str, expected: type[T]) -> T:
        # Type-asserts at the boundary so a typo / wrong-type clobber fails
        # loudly with a useful message instead of propagating None.
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
