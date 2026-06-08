"""typing.Protocol contracts for the modules runs.yaml references.

Duck-typed, your module doesn't import or inherit anything. Just
exporting the required attributes satisfies the Protocol. Static
checkers (Pyright/mypy) catch missing exports at edit time instead of
runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .context import Context
    from .specs import DatasetSpec, TrainedModel


@runtime_checkable
class DatasetModule(Protocol):
    """pipeline/datasets/<name>/, must export:"""
    NAME: str                                          # short slug, e.g. "fer2013"
    CLASS_NAMES: list[str]                             # index = int label
    def prepare(self, ctx: "Context") -> "DatasetSpec": ...   # acquire + split + persist


@runtime_checkable
class ModelModule(Protocol):
    """pipeline/models/<name>/, must export train(); build() + PREPROCESS optional."""
    def train(self, ctx: "Context", dataset: "DatasetSpec") -> "TrainedModel": ...


@runtime_checkable
class ConfigModule(Protocol):
    """configs/<name>.py, must export:"""
    NAME: str                                          # slug, e.g. "thorough"
    CONFIG: dict[str, Any]                             # hyperparameters
