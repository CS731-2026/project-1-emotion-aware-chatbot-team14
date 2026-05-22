from __future__ import annotations

from typing import Any


class Store:
    """In-memory key-value store for passing data between steps.

    Raises KeyError on missing keys — fail loudly, not silently.
    Silent overwrite on put — steps own their keys by convention.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        if key not in self._data:
            raise KeyError(
                f"store key {key!r} not found. "
                f"Available keys: {sorted(self._data.keys())}"
            )
        return self._data[key]

    def put(self, key: str, value: Any) -> None:
        self._data[key] = value

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def clear(self) -> None:
        self._data.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        keys = sorted(self._data.keys())
        return f"Store({keys})"
