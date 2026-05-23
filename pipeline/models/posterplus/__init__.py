"""POSTER++ — vendored as a sibling repo via git-weave.

Source of truth: Notebooks/3_benchmark_posterplus.ipynb. The notebook
clones github.com/Talented-Q/POSTER_V2 at run time; we do the same
via a `.thread` file at `vendor/POSTER_V2.thread` (weave config
`weave.json` scans `vendor/`). `make install-training` runs
`npx weave sync` to clone it into `vendor/POSTER_V2/` and then
symlinks that into `pipeline/models/posterplus/POSTER_V2/` so the
Python imports below resolve.

Architecture:
  - POSTER_V2's pyramid_trans_expr2 (Pyramid Transformer Expression v2)
  - 7-class classifier by default (RAF-DB convention); we override
    to dataset.num_classes for our 6-class EmpathBot use

First-run setup:
  make install-training   # pip install + weave sync + copy vendor tree
  # OR manually:
  npx weave sync
  cp -R vendor/POSTER_V2 pipeline/models/posterplus/POSTER_V2

The notebook is **inference-only** — it loads the published RAF-DB
checkpoint and reports accuracy / per-class metrics against the test
split. `train()` mirrors that: if the published checkpoint is found
(see inference_loop.py for resolution order), it runs the faithful
benchmark; otherwise it falls back to a generic fine-tune via
train_classifier.

To run the faithful benchmark, drop the checkpoint at
output/models/posterv2_rafdb.pth or set POSTER_CHECKPOINT_PATH (URL
in notebook cell 9).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torchvision.transforms as T

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.standard import train_classifier

from .inference_loop import run as _benchmark


_REPO_DIR = Path(__file__).parent / "POSTER_V2"

_IMG_SIZE = 224
PREPROCESS = T.Compose([
    T.Resize((_IMG_SIZE, _IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _import_poster():
    """Defer the POSTER_V2 import to build time so a missing clone
    only breaks runs that actually want this model — the rest of
    the pipeline keeps working."""
    if not _REPO_DIR.is_dir():
        raise FileNotFoundError(
            f"POSTER_V2 repo not found at {_REPO_DIR}. Run `make init` "
            "(or `npx weave sync`) to clone it via the .thread file."
        )
    if str(_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(_REPO_DIR))
    from models.PosterV2_7cls import pyramid_trans_expr2
    return pyramid_trans_expr2


def build(num_classes: int):
    """Instantiate POSTER++ for `num_classes`. The notebook uses 7
    (RAF-DB); for EmpathBot 6-class we pass 6 — POSTER++'s constructor
    accepts arbitrary num_classes."""
    pyramid_trans_expr2 = _import_poster()
    return pyramid_trans_expr2(img_size=_IMG_SIZE, num_classes=num_classes)


def _published_checkpoint_present(ctx: Context) -> bool:
    cfg = ctx.config.train_cfg
    for src in (cfg.get("checkpoint_path"),
                os.environ.get("POSTER_CHECKPOINT_PATH"),
                "output/models/posterv2_rafdb.pth"):
        if src and Path(src).exists():
            return True
    return False


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    """If the published RAF-DB checkpoint is on disk, run the faithful
    notebook-3 inference benchmark; otherwise fine-tune from ImageNet
    init via train_classifier."""
    model = build(dataset.num_classes)
    if _published_checkpoint_present(ctx):
        return _benchmark(ctx, dataset, model=model, preprocess=PREPROCESS)
    return train_classifier(ctx, dataset, model=model, preprocess=PREPROCESS)
