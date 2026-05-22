"""MLP classifier pipeline — model, steps, and epoch loop.

All pipeline logic lives here. Import what you need in run.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import harness
from harness import Success
from harness.store import Store


# ---------------------------------------------------------------------------
# Store keys — define once, reference everywhere; no magic strings in steps
# ---------------------------------------------------------------------------

DEVICE        = "device"
MODEL         = "model"
OPTIMIZER     = "optimizer"
TRAIN_DATASET = "train_dataset"
VAL_DATASET   = "val_dataset"
FOLDERS       = "folders"
LOGS          = "logs"
EPOCH         = "epoch"
TRAIN_LOSS    = "train_loss"
TRAIN_ACC     = "train_acc"
VAL_LOSS      = "val_loss"
VAL_ACC       = "val_acc"
BEST_VAL_ACC  = "best_val_acc"
FINAL_VAL_ACC = "final_val_acc"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Two-layer MLP: input_dim → hidden_dim (ReLU) → num_classes (logits)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def setup_device(store: Store, config: dict):
    # ── config ──────────────────────────────────────────────────────────────
    device_cfg = config.get("device", "auto")
    # ────────────────────────────────────────────────────────────────────────

    if device_cfg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_cfg

    store.put(DEVICE, device)
    print(f"         device: {device}")
    return Success


def setup_outputs(store: Store, config: dict):
    # ── store ────────────────────────────────────────────────────────────────
    run_dir = store.get("run_dir")
    # ─────────────────────────────────────────────────────────────────────────

    folders = harness.make_folders(run_dir, {
        "checkpoints": "checkpoints",
        "artifacts":   "artifacts",
        "eval":        "eval",
    })
    store.put(FOLDERS, folders)

    logs = harness.make_log_channels(run_dir, {
        "training": "logs/training.log",
        "eval":     "logs/eval.log",
    })
    store.put(LOGS, logs)
    return Success


def load_data(store: Store, config: dict):
    """Generate synthetic gaussian blobs; put TensorDatasets in store.

    DataLoaders are never stored — each training step constructs its own.
    """
    # ── config ───────────────────────────────────────────────────────────────
    input_dim   = config["input_dim"]
    num_classes = config["num_classes"]
    n_train     = config["n_train"]
    n_val       = config["n_val"]
    seed        = config.get("seed", 42)
    # ─────────────────────────────────────────────────────────────────────────

    rng = torch.Generator().manual_seed(seed)

    def _make_blobs(n: int) -> TensorDataset:
        torch.manual_seed(seed)
        centres = torch.randn(num_classes, input_dim)
        labels  = torch.randint(0, num_classes, (n,), generator=rng)
        noise   = torch.randn(n, input_dim, generator=rng) * 0.5
        x = centres[labels] + noise
        return TensorDataset(x, labels)

    store.put(TRAIN_DATASET, _make_blobs(n_train))
    store.put(VAL_DATASET,   _make_blobs(n_val))
    print(f"         train: {n_train} samples, val: {n_val} samples")
    return Success


def build_model(store: Store, config: dict):
    # ── config ───────────────────────────────────────────────────────────────
    input_dim   = config["input_dim"]
    hidden_dim  = config["hidden_dim"]
    num_classes = config["num_classes"]
    # ── store ────────────────────────────────────────────────────────────────
    device = store.get(DEVICE)
    # ─────────────────────────────────────────────────────────────────────────

    model = MLP(input_dim, hidden_dim, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"         MLP parameters: {n_params:,}")
    store.put(MODEL, model)
    return Success


def setup_optimizer(store: Store, config: dict):
    # ── config ───────────────────────────────────────────────────────────────
    lr = config["lr"]
    # ── store ────────────────────────────────────────────────────────────────
    model = store.get(MODEL)
    # ─────────────────────────────────────────────────────────────────────────

    store.put(OPTIMIZER, torch.optim.Adam(model.parameters(), lr=lr))
    return Success


def train_epoch(store: Store, config: dict):
    # ── config ───────────────────────────────────────────────────────────────
    batch_size = config["batch_size"]
    # ── store ────────────────────────────────────────────────────────────────
    model     = store.get(MODEL)
    optimizer = store.get(OPTIMIZER)
    device    = store.get(DEVICE)
    epoch     = store.get(EPOCH)
    logs      = store.get(LOGS)
    dataset   = store.get(TRAIN_DATASET)
    # ─────────────────────────────────────────────────────────────────────────

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss, all_logits, all_labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = classification_loss(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_logits.append(logits.detach())
        all_labels.append(y)

    avg_loss  = total_loss / len(loader)
    train_acc = accuracy(torch.cat(all_logits), torch.cat(all_labels))
    logs.training(f"epoch {epoch:03d}  train  loss={avg_loss:.4f}  acc={train_acc:.3f}")
    store.put(TRAIN_LOSS, avg_loss)
    store.put(TRAIN_ACC,  train_acc)
    return Success


def validate(store: Store, config: dict):
    # ── config ───────────────────────────────────────────────────────────────
    batch_size = config["batch_size"]
    # ── store ────────────────────────────────────────────────────────────────
    model   = store.get(MODEL)
    device  = store.get(DEVICE)
    epoch   = store.get(EPOCH)
    logs    = store.get(LOGS)
    dataset = store.get(VAL_DATASET)
    # ─────────────────────────────────────────────────────────────────────────

    loader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += classification_loss(logits, y).item()
            all_logits.append(logits)
            all_labels.append(y)

    avg_loss = total_loss / len(loader)
    val_acc  = accuracy(torch.cat(all_logits), torch.cat(all_labels))
    logs.training(f"epoch {epoch:03d}  val    loss={avg_loss:.4f}  acc={val_acc:.3f}")
    store.put(VAL_LOSS, avg_loss)
    store.put(VAL_ACC,  val_acc)
    return Success


def save_best_checkpoint(store: Store, config: dict):
    # ── store ────────────────────────────────────────────────────────────────
    model   = store.get(MODEL)
    folders = store.get(FOLDERS)
    val_acc = store.get(VAL_ACC)
    epoch   = store.get(EPOCH)
    best    = store.get(BEST_VAL_ACC) if BEST_VAL_ACC in store else -1.0
    # ─────────────────────────────────────────────────────────────────────────

    if val_acc > best:
        store.put(BEST_VAL_ACC, val_acc)
        folders.save("checkpoints", "best", model.state_dict())
        store.get(LOGS).training(
            f"epoch {epoch:03d}  new best val_acc={val_acc:.3f}  checkpoint saved"
        )
    return Success


def evaluate(store: Store, config: dict):
    # ── config ───────────────────────────────────────────────────────────────
    input_dim   = config["input_dim"]
    hidden_dim  = config["hidden_dim"]
    num_classes = config["num_classes"]
    batch_size  = config["batch_size"]
    # ── store ────────────────────────────────────────────────────────────────
    device  = store.get(DEVICE)
    folders = store.get(FOLDERS)
    logs    = store.get(LOGS)
    dataset = store.get(VAL_DATASET)
    # ─────────────────────────────────────────────────────────────────────────

    model = MLP(input_dim, hidden_dim, num_classes).to(device)
    model.load_state_dict(folders.load("checkpoints", "best"))
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            all_logits.append(model(x))
            all_labels.append(y)

    final_acc = accuracy(torch.cat(all_logits), torch.cat(all_labels))
    logs.eval(f"final val accuracy (best checkpoint): {final_acc:.4f}")
    store.put(FINAL_VAL_ACC, final_acc)
    print(f"         final val accuracy: {final_acc:.4f}")
    return Success


def save_summary(store: Store, config: dict):
    # ── config ───────────────────────────────────────────────────────────────
    epochs = config["epochs"]
    # ── store ────────────────────────────────────────────────────────────────
    folders   = store.get(FOLDERS)
    final_acc = store.get(FINAL_VAL_ACC)
    best_acc  = store.get(BEST_VAL_ACC) if BEST_VAL_ACC in store else None
    # ─────────────────────────────────────────────────────────────────────────

    folders.save("artifacts", "summary", {
        "final_val_acc":  final_acc,
        "best_val_acc":   best_acc,
        "epochs_trained": epochs,
        "config":         config,
    })
    print("         summary saved to artifacts/")
    return Success
