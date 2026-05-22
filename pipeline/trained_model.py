"""TrainedModel — the typed handoff produced by the train phase.

Holds the bookkeeping that downstream phases (a future dedicated
evaluate phase) need to load and use the checkpoint without
re-deriving anything: where the weights live, which model module
built them, the per-epoch metrics history, and the final val numbers.

The checkpoint file itself lives under the run dir's checkpoints/
subdir; this struct just records the path so reloading is one
torch.load away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrainedModel:
    model_name:       str                # e.g. "tiny_cnn" — for importlib.import_module
    num_classes:      int
    checkpoint_path:  Path
    history:          list[dict] = field(default_factory=list)  # one dict per epoch
    final_val:        dict = field(default_factory=dict)        # {"loss": ..., "acc": ...}
    final_test:       dict = field(default_factory=dict)        # filled if test eval ran
