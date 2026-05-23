"""POSTER++ — vendored as a sibling repo via git-weave.

Source of truth: Notebooks/3_benchmark_posterplus.ipynb. The notebook
clones github.com/Talented-Q/POSTER_V2 at run time; we do the same
via a `.thread` file (see POSTER_V2.thread) so `make init` /
`npx weave sync` pulls it into pipeline/models/posterplus/POSTER_V2/
on first setup.

Architecture:
  - POSTER_V2's pyramid_trans_expr2 (Pyramid Transformer Expression v2)
  - 7-class classifier by default (RAF-DB convention); we override
    to dataset.num_classes for our 6-class EmpathBot use

First-run setup:
  make init          # → `npx weave sync` clones POSTER_V2 here
  # OR manually:
  git clone https://github.com/Talented-Q/POSTER_V2.git \\
      pipeline/models/posterplus/POSTER_V2

If you want to load the official pretrained weights from the paper,
drop the checkpoint at output/models/posterv2_rafdb.pth and uncomment
the load_state_dict line in train() (notebook cell 13 has the
checkpoint URL).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torchvision.transforms as T

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.standard import train_classifier


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


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    return train_classifier(
        ctx, dataset,
        model=build(dataset.num_classes),
        preprocess=PREPROCESS,
    )
