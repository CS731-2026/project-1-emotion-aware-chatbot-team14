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

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from pipeline.context import Context


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
