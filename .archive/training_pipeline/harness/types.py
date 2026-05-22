from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union


class _Success:
    """Sentinel returned by a step to signal successful completion."""

    def __repr__(self) -> str:
        return "Success"

    def __bool__(self) -> bool:
        return True


Success = _Success()


@dataclass
class Failure:
    """Returned by a step to signal a controlled failure."""

    reason: str

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"Failure({self.reason!r})"


@dataclass
class Loop:
    """A sequence of steps repeated for up to n iterations.

    Created via harness.loop(). The harness owns serialisation and folder
    creation at every level — the pipeline only defines the steps and conditions.

    Args:
        steps:     Child steps (callables or nested Loop objects).
        n:         Maximum iterations. Int or callable (config) -> int.
        iter_name: Name of the loop variable. Used for folder names and the
                   store key injected before each iteration (e.g. "epoch"
                   → folders epoch_000/, store["epoch"] = 0).
        while_:    Optional continue predicate (store, config) -> bool.
                   Evaluated after each completed iteration using the live
                   store. If it returns False the loop exits early.
    """

    steps: list
    n: Union[int, Callable]
    iter_name: str = "iter"
    while_: Union[Callable, None] = None

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return f"{self.iter_name}_loop"
