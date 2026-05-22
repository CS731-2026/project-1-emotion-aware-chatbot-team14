"""POSTER++ — external-repo model. Pending vendor-in.

Source of truth: Notebooks/3_benchmark_posterplus.ipynb. The notebook
clones the official POSTER_V2 repo at runtime and imports
`models.PosterV2_7cls.pyramid_trans_expr2` from it. We can't lift the
architecture into this package without vendoring that repo (or
referencing it as a submodule).

To make this runnable:
  1. Clone POSTER_V2 to a known location (the notebook uses
     `POSTER_REPO_DIR` env var)
  2. Add it to PYTHONPATH or vendor `models/PosterV2_7cls.py` into
     `pipeline/models/posterplus/poster_v2.py`
  3. Replace this stub's `build()` with a real architecture call
  4. Decide where the official checkpoint lives — model_service/models/
     or output/models/

Until then, `train()` raises with setup instructions.
"""

from __future__ import annotations

import torchvision.transforms as T

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel


_IMG_SIZE = 224
PREPROCESS = T.Compose([
    T.Resize((_IMG_SIZE, _IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    raise NotImplementedError(
        "POSTER++ depends on the external POSTER_V2 repo (see "
        "Notebooks/3_benchmark_posterplus.ipynb cell 4). Vendor in "
        "models/PosterV2_7cls.py and wire build() before adding this "
        "model to pipeline/train.py's RUNS list."
    )
