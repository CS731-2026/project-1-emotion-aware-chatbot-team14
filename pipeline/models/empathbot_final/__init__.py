"""EmpathBot final (v4), faithful port of
Notebooks/5_final_empathbot_training_v4.ipynb.

Sibling to empathbot_v1 (notebook 6b). Same class name in both
notebooks (`EmpathBotV1`) but different code paths, 6b uses timm;
5_v4 uses torchvision with selectable backbone (efficientnet_b2 |
resnet18) + manual SE for the resnet18 variant.

Package layout, each file mirrors one section of the notebook:

  __init__.py     pipeline surface (PREPROCESS + train)
  model.py        architecture (cell 16) + freeze/unfreeze methods (cell 22)
  augment.py      STD_TF / NEG_TF / VAL_TF (cell 14) + NEGATIVE_LABEL_IDS
  data.py         EmpathBotDataset with NEG/STD routing (cell 14)
  train_loop.py   MixUp + freeze schedule + split-LR AdamW + cosine
                  LambdaLR + label-smoothed weighted CE (cells 20, 22, 24, 25)

Checkpoint envelope matches notebook cell 25:
  epoch, val_acc, model_state_dict, optimizer_state_dict,
  empathbot_classes, architecture
"""

from __future__ import annotations

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel

from .augment import VAL_TF as PREPROCESS
from .model import build
from .train_loop import run as _run


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    return _run(ctx, dataset, model=build(dataset.num_classes))
