"""Typing.Protocol contracts for the modules referenced from runs.yaml.

These make the implicit "every dataset/model module must export X"
convention explicit + checkable. Pyright / mypy will flag a model
module that forgets to export `train`, or a dataset module that
returns the wrong shape from `prepare`, at edit time — before
`make train` is ever invoked.

The framework code (driver, phases) types its module references with
these Protocols. Your model + dataset modules don't have to import or
inherit anything — duck typing means just exporting the required
attributes is enough. The Protocol is documentation + a static check.

Read the model/dataset tutorials (pipeline/models/tutorial/,
pipeline/datasets/tutorial/) for a working example of each.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .context import Context
    from .specs import DatasetSpec, TrainedModel


@runtime_checkable
class DatasetModule(Protocol):
    """Contract every pipeline/datasets/<name>/ package satisfies.

    Required exports:
      NAME           short slug used in run dirs (e.g. "fer2013")
      CLASS_NAMES    target classes the dataset emits, index = int label
      prepare(ctx)   acquire + remap + split + persist, return DatasetSpec
    """

    NAME: str
    CLASS_NAMES: list[str]

    def prepare(self, ctx: "Context") -> "DatasetSpec":
        """Build the dataset (cache-aware) and return a DatasetSpec
        pointing at the train/val/test CSVs."""
        ...


@runtime_checkable
class ModelModule(Protocol):
    """Contract every pipeline/models/<name>/ package satisfies.

    Required:
      train(ctx, dataset)   train the model, return a TrainedModel

    Optional but conventional (used by deploy + the live model_service):
      build(num_classes)    factory returning a fresh nn.Module
      PREPROCESS            torchvision transform used at inference time
    """

    def train(self, ctx: "Context", dataset: "DatasetSpec") -> "TrainedModel":
        """Train the model, write artifacts + checkpoints into the run
        dir via ctx.save_*, return a TrainedModel."""
        ...


@runtime_checkable
class ConfigModule(Protocol):
    """Contract every configs/<name>.py satisfies.

    Required exports:
      NAME       short slug used in run dirs (e.g. "thorough")
      CONFIG     dict of hyperparameters; shallow-merged over each model's CFG
    """

    NAME: str
    CONFIG: dict[str, Any]
