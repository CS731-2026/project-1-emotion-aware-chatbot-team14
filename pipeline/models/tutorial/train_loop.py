"""train_loop.py — the actual training. CFG + loop + reporting.

This is the meat of a model module. Everything else (model.py,
augment.py, data.py) is just wired in here. Read this file last.

The structure mirrors what your notebook probably had:
  1. CFG = hyperparameters (defaults)
  2. Inner helpers: train one epoch, eval one epoch
  3. run(ctx, dataset, model) — orchestrates the whole training:
       a. resolve hparams (defaults + overrides)
       b. build the dataloaders
       c. build loss + optimizer + scheduler
       d. loop: train_epoch + eval_epoch + save best checkpoint
       e. final test eval + write artifacts
       f. return TrainedModel for the driver

The framework provides helpers (`pipeline.training.loop`,
`pipeline.training.reporting`) — use them where the boilerplate is
worth saving, hand-roll where you need control.

KEY THINGS THIS DEMONSTRATES:
  - merge_cfg                       hparam resolution + auto logging
  - ctx.save_scalar                 per-epoch metric → metrics.jsonl
  - ctx.save_checkpoint             best.pth (use this; not torch.save)
  - ctx.save_image                  custom matplotlib plot (extra art)
  - ctx.save_json                   any structured output
  - ctx.save_text                   plain-text notes / debug dumps
  - collect_predictions             one walk over test_loader → preds + labels
  - write_standard_artifacts        the 4 standard artifacts in one call
  - early stopping on val_acc
  - class weights from dataset.class_weights
  - device selection via auto_device
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Framework imports — these are the helpers worth knowing.
#   - Context, DatasetSpec, TrainedModel   typed contracts with the driver
#   - auto_device                          CUDA → MPS → CPU
#   - collect_predictions                  test-pass preds/labels for reporting
#   - merge_cfg                            CFG + overrides + auto-log
#   - write_standard_artifacts             one call → 4 artifact files
from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.loop import auto_device, collect_predictions, merge_cfg
from pipeline.training.reporting import write_standard_artifacts

# Sibling files.
from .augment import TRAIN_TF, VAL_TF
from .data import CsvImageDataset

# Convention: every model gets its own logger. Logs land in stdout
# automatically when the driver sets up logging.
logger = logging.getLogger(__name__)


# ── CFG — default hyperparameters ────────────────────────────────────────
#
# This dict is the SINGLE place hyperparameters live for this model.
# Three things override these at runtime, in order:
#
#   1. configs/<name>.py::CONFIG     (fast / baseline / thorough)
#   2. runs.yaml's per-run train_cfg block
#
# Any key in CFG is overridable. Add a new key here and it's
# automatically in scope — no whitelist to maintain (see merge_cfg).
#
# Values you set here should be the "if you had to pick one number
# that worked, what would it be" defaults — usually whatever your
# notebook converged on.
CFG = dict(
    # ── core training schedule ─────────────────
    epochs       = 30,
    batch_size   = 32,
    num_workers  = 0,        # 0 = main process. 2-4 on real datasets if you have RAM.

    # ── optimiser ──────────────────────────────
    lr           = 1.0e-4,   # safe for fine-tuning a pretrained backbone
    weight_decay = 1.0e-4,   # L2 reg; helps when overfitting
    momentum     = 0.9,      # only used by SGD; here for the example below

    # ── early stop ─────────────────────────────
    early_stop   = 10,       # epochs without val_acc improvement before bailing
)


# ── inner helpers (one epoch of train / eval) ────────────────────────────

def _train_one_epoch(model, loader, criterion, optimizer, device):
    """Standard supervised training loop, one pass over `loader`.

    Note: keep this small. Anything fancier (MixUp, gradient
    accumulation, mixed-precision) lives in its own helper that this
    function calls. Don't grow this past 30 lines.
    """
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * imgs.size(0)
        correct += int(out.argmax(1).eq(labels).sum().item())
        total += int(labels.size(0))
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def _eval_one_epoch(model, loader, criterion, device):
    """Same structure as train_one_epoch but no gradients and no
    optimiser step. Returns (loss, acc)."""
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += float(loss.item()) * imgs.size(0)
        correct += int(out.argmax(1).eq(labels).sum().item())
        total += int(labels.size(0))
    return total_loss / max(1, total), correct / max(1, total)


# ── the main event ───────────────────────────────────────────────────────

def run(ctx: Context, dataset: DatasetSpec, model: nn.Module) -> TrainedModel:
    """Train `model` on `dataset` using ctx.config.train_cfg.

    The driver calls this via __init__.py::train(). All training logic
    lives here. Return a TrainedModel — the driver uses its
    checkpoint_path + final_test metrics for leaderboard / deploy.
    """
    # ── 1. Resolve hyperparameters ─────────────────────────────────
    # merge_cfg does two things:
    #   - returns {**CFG, **(overrides ∩ CFG keys)} (shallow merge)
    #   - logs three INFO lines: defaults / overrides / resolved
    # So your run's stdout self-documents what config it ran with.
    # The "ignored unknown keys" warning catches typos and
    # cross-model keys silently.
    cfg = merge_cfg(CFG, ctx.config.train_cfg)
    device = auto_device()
    model = model.to(device)
    num_classes = dataset.num_classes

    logger.info("tutorial: device=%s epochs=%d batch=%d lr=%.1e",
                device, cfg["epochs"], cfg["batch_size"], cfg["lr"])

    # ── 2. DataLoaders ─────────────────────────────────────────────
    # dataset.splits is a dict {"train": Path, "val": Path, "test": Path}.
    # The dataset module already split and saved the CSVs; you just
    # wrap each in a Dataset + DataLoader. The pin_memory flag helps
    # when you're on CUDA; harmless on CPU/MPS.
    pin = device.type == "cuda"
    train_loader = DataLoader(
        CsvImageDataset(dataset.splits["train"], TRAIN_TF),
        batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=pin,
    )
    val_loader = DataLoader(
        CsvImageDataset(dataset.splits["val"], VAL_TF),
        batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=pin,
    )
    test_loader = DataLoader(
        CsvImageDataset(dataset.splits["test"], VAL_TF),
        batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=pin,
    )

    # ── 3. Loss, optimiser, scheduler ──────────────────────────────
    #
    # Class weights: when the dataset is imbalanced, weighted CE
    # punishes mistakes on minority classes more heavily so the model
    # doesn't just predict the majority class. The dataset module
    # computes these (inverse frequency); we just pass them through.
    weight = None
    if dataset.class_weights is not None:
        weight = torch.tensor(dataset.class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Optimiser: AdamW is a safe default for fine-tuning. Switch to
    # SGD(momentum=0.9) for from-scratch training of conv nets;
    # AdamW for almost everything else.
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )

    # Scheduler: drop LR by half every 10 epochs. Cosine annealing
    # (CosineAnnealingLR / LambdaLR with cosine) often works better;
    # we use StepLR here for simplicity.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ── 4. Training loop ───────────────────────────────────────────
    history: list[dict] = []
    best_val_acc = 0.0
    best_epoch = 0
    patience = 0

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = _eval_one_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # ── ctx.save_scalar: appends one line to metrics.jsonl ────
        # Cheap (plain text append) — call it for every metric you
        # want to plot later. The `step` arg becomes the x-axis when
        # someone reads metrics.jsonl back into pandas.
        ctx.save_scalar("train/loss", tr_loss, step=epoch - 1)
        ctx.save_scalar("train/acc",  tr_acc,  step=epoch - 1)
        ctx.save_scalar("val/loss",   vl_loss, step=epoch - 1)
        ctx.save_scalar("val/acc",    vl_acc,  step=epoch - 1)
        ctx.save_scalar("lr",         current_lr, step=epoch - 1)

        # Keep a Python-side history too — feeds the training_curves
        # PNG that write_standard_artifacts produces.
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": vl_loss, "val_acc": vl_acc, "lr": current_lr,
        })
        logger.info(
            "epoch %d/%d  tr_loss=%.4f tr_acc=%.3f  vl_loss=%.4f vl_acc=%.3f  lr=%.1e",
            epoch, cfg["epochs"], tr_loss, tr_acc, vl_loss, vl_acc, current_lr,
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch = epoch
            patience = 0
            # ── ctx.save_checkpoint: writes to checkpoints/<name>.pth
            # The state dict envelope is up to you — include whatever
            # you'd need to reload the model for inference + debug.
            # Don't include massive things you can re-derive (the dataset).
            ctx.save_checkpoint("best", {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": vl_acc,
                "val_loss": vl_loss,
                "class_names": dataset.class_names,
            })
        else:
            patience += 1
            if patience >= cfg["early_stop"]:
                logger.info("early stop at epoch %d (best val_acc=%.4f @ %d)",
                            epoch, best_val_acc, best_epoch)
                break

    # ── 5. Final test evaluation on the best checkpoint ────────────
    #
    # Reload best weights — early stop or natural overfitting might
    # mean the LAST epoch's model isn't the BEST. We want test metrics
    # measured against the checkpoint we'd actually ship.
    best_ckpt = ctx.run_dir / "checkpoints" / "best.pth"
    if best_ckpt.exists():
        ck = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])

    test_loss, test_acc = _eval_one_epoch(model, test_loader, criterion, device)

    # ── collect_predictions: ONE walk over the test loader returning
    # (preds, labels) as flat Python lists. We pass these to
    # write_standard_artifacts so it can build the confusion matrix +
    # classification report without re-walking the loader.
    test_preds, test_labels = collect_predictions(model, test_loader, device)

    ctx.save_scalar("test/loss", test_loss)
    ctx.save_scalar("test/acc",  test_acc)

    # ── ctx.save_json: any JSON-serialisable dict. Goes to
    # artifacts/history.json. The standard reporting helper writes
    # final.json separately, but you can save your own custom JSONs
    # for whatever extra analysis you want.
    ctx.save_json("history", history)

    # ── write_standard_artifacts: one call → four files
    #   artifacts/training_curves.png       train/val loss + acc + lr
    #   artifacts/confusion_matrix.png      raw + normalised
    #   artifacts/classification_report.txt sklearn per-class P/R/F1
    #   artifacts/per_class_metrics.json    structured
    # Plus final.json with the summary dict you pass.
    # If you want extra plots beyond these, use ctx.save_image directly
    # (see § "custom artifact" below for an example).
    write_standard_artifacts(
        ctx,
        history=history,
        test_preds=test_preds,
        test_labels=test_labels,
        num_classes=num_classes,
        class_names=dataset.class_names,
        final_summary={
            "best_epoch":   best_epoch,
            "best_val_acc": best_val_acc,
            "test_acc":     test_acc,
            "test_loss":    test_loss,
        },
    )

    # ── custom artifact example: per-class confidence distribution ─
    # Anything you can make with matplotlib lands in artifacts/ via
    # ctx.save_image. The run dir is abstracted — you only name the
    # file, not the path.
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend; works without a display
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(range(num_classes), [test_labels.count(i) for i in range(num_classes)])
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(dataset.class_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("test set count")
        ax.set_title(f"{ctx.config.model} — test class distribution")
        ctx.save_image("test_class_distribution", fig)
        plt.close(fig)
    except Exception as e:
        # Custom artifacts shouldn't ever fail the run — log and continue.
        logger.warning("tutorial: custom artifact skipped (%s)", e)

    # ── ctx.save_text: plain text. Use for human-readable summaries,
    # prompts, debug dumps, anything you'd otherwise tail in stdout
    # and lose.
    ctx.save_text("notes", (
        f"tutorial run complete.\n"
        f"  best val acc {best_val_acc:.4f} @ epoch {best_epoch}\n"
        f"  test acc      {test_acc:.4f}\n"
        f"  device        {device}\n"
    ))

    # ── 6. Return TrainedModel — the driver's contract ─────────────
    # The leaderboard, deploy step, and audit tooling all read these
    # fields. Don't skip any.
    return TrainedModel(
        model_name=ctx.config.model,
        num_classes=num_classes,
        checkpoint_path=best_ckpt,
        history=history,
        final_val={"acc": best_val_acc},
        final_test={"acc": test_acc, "loss": test_loss},
    )
