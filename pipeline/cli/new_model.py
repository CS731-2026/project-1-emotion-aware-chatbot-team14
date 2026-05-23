"""Scaffold a new model module under pipeline/models/<id>/.

Two templates:
  --template simple   30-line minimum using train_classifier
  --template custom   full 6-file layout (model + augment + data + loss + train_loop)
                      with sensible defaults copied from empathbot_v1

Usage:
    python -m pipeline.cli.new_model my_model --template simple
    python -m pipeline.cli.new_model my_model --template custom

After generation, add a line to runs.yaml:
    runs:
      - { dataset: synthetic_smoke, model: my_model, config: fast }
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SIMPLE_INIT = '''\
"""{model_id} — simple classifier using the shared train_classifier helper.

For a custom training procedure (split-LR, freeze schedules, MixUp,
focal loss, etc) scaffold with --template custom instead, or look at
pipeline/models/empathbot_v1/ for a full-featured example.
"""

from __future__ import annotations

import torch.nn as nn

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.standard import train_classifier

from .augment import PREPROCESS
from .model import build


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    return train_classifier(
        ctx, dataset,
        model=build(dataset.num_classes),
        preprocess=PREPROCESS,
    )
'''

SIMPLE_MODEL = '''\
"""Architecture for {model_id}. Edit me."""

from __future__ import annotations

import torch.nn as nn


class {class_name}(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # TODO: replace with your real architecture
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build(num_classes: int) -> nn.Module:
    return {class_name}(num_classes=num_classes)
'''

SIMPLE_AUGMENT = '''\
"""Transforms for {model_id}. PREPROCESS is what train_classifier
prepends a configurable augmentation block to (see configs/*.py)."""

from __future__ import annotations

import torchvision.transforms as T


IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


PREPROCESS = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
'''

CUSTOM_INIT = '''\
"""{model_id} — full custom training loop.

  __init__.py     pipeline surface (build + train)
  model.py        nn.Module class
  augment.py      TRAIN_TF / VAL_TF transforms
  data.py         Dataset class with any per-sample routing
  train_loop.py   CFG + run() that orchestrates training
"""

from __future__ import annotations

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel

from .augment import VAL_TF as PREPROCESS
from .model import build
from .train_loop import run as _run


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    return _run(ctx, dataset, model=build(dataset.num_classes))
'''

CUSTOM_MODEL = '''\
"""Architecture for {model_id}. Edit me."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class {class_name}(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        bb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = bb.fc.in_features
        bb.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        self.net = bb

    def forward(self, x):
        return self.net(x)


def build(num_classes: int) -> nn.Module:
    return {class_name}(num_classes=num_classes)
'''

CUSTOM_AUGMENT = '''\
"""Train + val transforms for {model_id}."""

from __future__ import annotations

import torchvision.transforms as T


IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


TRAIN_TF = T.Compose([
    T.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

VAL_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
'''

CUSTOM_DATA = '''\
"""Dataset class for {model_id}.

For most cases the train_loop can read directly from a CSV via a
simple wrapper — see pipeline/models/resnet18/train_loop.py's
_CsvDataset for the minimal pattern. Add a custom class here only if
you need per-sample routing (e.g. hard classes get stronger
augmentation, see pipeline/models/empathbot_v1/data.py)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CsvImageDataset(Dataset):
    def __init__(self, csv_path, transform) -> None:
        df = pd.read_csv(csv_path)
        valid = df["path"].apply(lambda p: Path(p).exists())
        self.df = df[valid].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), int(row["label"])
'''

CUSTOM_TRAIN_LOOP = '''\
"""Training loop for {model_id}. Edit CFG to tweak defaults."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.loop import auto_device, collect_predictions, merge_cfg
from pipeline.training.reporting import write_standard_artifacts

from .augment import TRAIN_TF, VAL_TF
from .data import CsvImageDataset

logger = logging.getLogger(__name__)


# Defaults — overrideable from configs/*.py CONFIG or runs.yaml train_cfg.
# Any key here becomes overridable; no whitelist to maintain.
CFG = dict(
    epochs       = 20,
    batch_size   = 32,
    num_workers  = 0,
    lr           = 1.0e-4,
    weight_decay = 1.0e-4,
    early_stop   = 10,
)


def _run_epoch(model, loader, criterion, optimizer, device, *, training: bool):
    model.train() if training else model.eval()
    total_loss = correct = total = 0
    ctx_mgr = torch.enable_grad() if training else torch.no_grad()
    with ctx_mgr:
        for imgs, labels in loader:
            imgs = imgs.to(device); labels = labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += float(loss.item()) * imgs.size(0)
            correct += int(out.argmax(1).eq(labels).sum().item())
            total += int(labels.size(0))
    return total_loss / max(1, total), correct / max(1, total)


def run(ctx: Context, dataset: DatasetSpec, model: nn.Module) -> TrainedModel:
    cfg = merge_cfg(CFG, ctx.config.train_cfg)
    device = auto_device()
    model = model.to(device)
    num_classes = dataset.num_classes

    train_loader = DataLoader(CsvImageDataset(dataset.splits["train"], TRAIN_TF),
                               batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"])
    val_loader   = DataLoader(CsvImageDataset(dataset.splits["val"],   VAL_TF),
                               batch_size=cfg["batch_size"], shuffle=False,
                               num_workers=cfg["num_workers"])
    test_loader  = DataLoader(CsvImageDataset(dataset.splits["test"],  VAL_TF),
                               batch_size=cfg["batch_size"], shuffle=False,
                               num_workers=cfg["num_workers"])

    weight = None
    if dataset.class_weights is not None:
        weight = torch.tensor(dataset.class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    history = []
    best_val_acc = 0.0
    best_epoch = 0
    patience = 0
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        vl_loss, vl_acc = _run_epoch(model, val_loader,   criterion, optimizer, device, training=False)
        ctx.save_scalar("train/loss", tr_loss, step=epoch - 1)
        ctx.save_scalar("train/acc",  tr_acc,  step=epoch - 1)
        ctx.save_scalar("val/loss",   vl_loss, step=epoch - 1)
        ctx.save_scalar("val/acc",    vl_acc,  step=epoch - 1)
        history.append({{
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": vl_loss, "val_acc": vl_acc,
        }})
        logger.info("epoch %d/%d  tr_loss=%.4f tr_acc=%.3f  vl_loss=%.4f vl_acc=%.3f",
                    epoch, cfg["epochs"], tr_loss, tr_acc, vl_loss, vl_acc)
        if vl_acc > best_val_acc:
            best_val_acc, best_epoch, patience = vl_acc, epoch, 0
            ctx.save_checkpoint("best", {{"epoch": epoch, "val_acc": vl_acc,
                                          "model_state_dict": model.state_dict()}})
        else:
            patience += 1
            if patience >= cfg["early_stop"]:
                logger.info("early stop at epoch %d", epoch); break

    best_ckpt = ctx.run_dir / "checkpoints" / "best.pth"
    if best_ckpt.exists():
        ck = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
    test_loss, test_acc = _run_epoch(model, test_loader, criterion, optimizer, device, training=False)
    test_preds, test_labels = collect_predictions(model, test_loader, device)
    ctx.save_scalar("test/loss", test_loss); ctx.save_scalar("test/acc", test_acc)
    ctx.save_json("history", history)
    write_standard_artifacts(
        ctx, history=history,
        test_preds=test_preds, test_labels=test_labels,
        num_classes=num_classes, class_names=dataset.class_names,
        final_summary={{"best_epoch": best_epoch, "best_val_acc": best_val_acc,
                        "test_acc": test_acc, "test_loss": test_loss}},
    )
    return TrainedModel(
        model_name=ctx.config.model, num_classes=num_classes,
        checkpoint_path=best_ckpt, history=history,
        final_val={{"acc": best_val_acc}},
        final_test={{"acc": test_acc, "loss": test_loss}},
    )
'''


def _class_name_for(model_id: str) -> str:
    """my_model → MyModel."""
    return "".join(part.capitalize() for part in model_id.split("_"))


def _write(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists (pass --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_id",
                    help="snake_case model id; becomes pipeline/models/<id>/")
    ap.add_argument("--template", choices=("simple", "custom"), default="simple")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing files (default: fail if any exist)")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()

    model_id = args.model_id
    if not model_id.replace("_", "").isalnum() or not model_id[0].isalpha():
        print(f"✗ model_id must be snake_case ascii starting with a letter: {model_id!r}",
              file=sys.stderr)
        return 2

    repo_root = Path(args.repo_root).resolve()
    target = repo_root / "pipeline" / "models" / model_id
    if target.exists() and not args.force:
        print(f"✗ {target} already exists. Pass --force to overwrite, or pick a different id.",
              file=sys.stderr)
        return 2

    cls = _class_name_for(model_id)
    if args.template == "simple":
        files = {
            "__init__.py": SIMPLE_INIT.format(model_id=model_id),
            "model.py":    SIMPLE_MODEL.format(model_id=model_id, class_name=cls),
            "augment.py":  SIMPLE_AUGMENT.format(model_id=model_id),
        }
    else:
        files = {
            "__init__.py":   CUSTOM_INIT.format(model_id=model_id),
            "model.py":      CUSTOM_MODEL.format(model_id=model_id, class_name=cls),
            "augment.py":    CUSTOM_AUGMENT.format(model_id=model_id),
            "data.py":       CUSTOM_DATA.format(model_id=model_id),
            "train_loop.py": CUSTOM_TRAIN_LOOP.format(model_id=model_id),
        }

    for name, content in files.items():
        _write(target / name, content, args.force)
        print(f"  + {target.relative_to(repo_root)}/{name}")

    print()
    print(f"✓ scaffolded pipeline/models/{model_id}/ ({args.template} template)")
    print()
    print("Next:")
    print(f"  1. edit pipeline/models/{model_id}/model.py — your architecture")
    if args.template == "custom":
        print(f"  2. edit pipeline/models/{model_id}/train_loop.py CFG — hyperparameters")
    print("  3. add a line to runs.yaml:")
    print(f"       - {{ dataset: synthetic_smoke, model: {model_id}, config: fast }}")
    print(f"  4. make train RUN={model_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
