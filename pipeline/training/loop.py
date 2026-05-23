"""Inner training mechanics — train_one_epoch + evaluate.

These two functions own the per-step / per-batch detail so the train
phase stays orchestration. They're architecture-blind: any model that
takes batched RGB tensors and emits class-count logits runs here.

Metrics are kept minimal at this layer (mean batch loss + overall
accuracy). Richer per-class breakdowns + plots live in a later
evaluate phase and can be re-run post-hoc against the saved
checkpoint without retraining.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from pipeline.context import Context


def merge_cfg(default: dict, overrides: dict | None) -> dict:
    """Shallow-merge `overrides` over `default`, ignoring unknown keys,
    and log the resolved + diff so it's never ambiguous what
    hyperparameters a run actually used.

    Used by every model's `run(ctx, dataset, model)` to apply the
    pipeline-level CONFIG dict over the model's own CFG defaults:

        cfg = merge_cfg(CFG, ctx.config.train_cfg)

    Unknown keys in `overrides` (typos, keys that belong to a different
    model) are silently dropped so a single shared config can be passed
    to many models without each one breaking on extra fields. Any key
    in `default` is automatically overridable — no manual whitelist.

    Three lines land at INFO so the run log is self-documenting:
      hparams (defaults): {...}        ← model's CFG (notebook values)
      hparams (overrides): {...}       ← what the config/train_cfg supplied
      hparams (resolved): {...}        ← what training will actually use
    Plus a warning for any override keys that were ignored as unknown.
    """
    resolved = {**default, **{k: v for k, v in (overrides or {}).items()
                               if k in default}}

    _log = logging.getLogger("pipeline.training.hparams")
    if overrides:
        applied   = {k: v for k, v in overrides.items() if k in default}
        ignored   = {k: v for k, v in overrides.items() if k not in default}
        _log.info("hparams (defaults):  %s", default)
        _log.info("hparams (overrides): %s", applied if applied else "(none applied)")
        _log.info("hparams (resolved):  %s", resolved)
        if ignored:
            _log.warning(
                "hparams: %d override key(s) ignored (not in this model's CFG): %s",
                len(ignored), sorted(ignored),
            )
    else:
        _log.info("hparams (resolved):  %s", resolved)
    return resolved


def auto_device() -> torch.device:
    """Pick CUDA → MPS → CPU. Matches what core/face_detector and the
    emotion model do — keep selection consistent across the project."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    loss_fn:    nn.Module,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    epoch:      int,
    ctx:        "Context",
    log_every:  int = 20,
) -> dict[str, float]:
    """One epoch over `loader`. Returns {"loss": mean batch loss}. Logs
    per-batch loss every `log_every` steps via ctx.save_scalar so
    metrics.jsonl gets a usable curve without overwhelming the file."""
    model.train()
    running_loss = 0.0
    n_batches = 0
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        n_batches += 1

        if batch_idx % log_every == 0:
            step = epoch * len(loader) + batch_idx
            ctx.save_scalar("train/loss", float(loss.item()), step=step)

    return {"loss": running_loss / max(1, n_batches)}


@torch.no_grad()
def collect_predictions(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Run inference over `loader` and return (preds, labels) as flat
    Python lists. Used by the post-training reporting step so each
    model only needs to walk the test loader once for both accuracy
    metrics and per-class breakdowns."""
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        labels.extend(y.tolist())
    return preds, labels


@torch.no_grad()
def evaluate(
    model:    nn.Module,
    loader:   DataLoader,
    loss_fn:  nn.Module,
    device:   torch.device,
) -> dict[str, float]:
    """Run inference over `loader`, returning overall {loss, acc}.
    Single-number metrics only — split-aware logging is the caller's
    job (the train phase tags these as val/ or test/)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item())
        n_batches += 1

        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    return {
        "loss": total_loss / max(1, n_batches),
        "acc":  correct / max(1, total),
    }
