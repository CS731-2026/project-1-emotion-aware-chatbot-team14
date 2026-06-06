"""Phase functions. Each has signature (ctx: Context) -> None and is
registered in driver.py's PHASES dict; the driver iterates Context.config.phases."""

from __future__ import annotations

import logging
import random

from .framework import keys as K
from .framework.context import Context
from .framework.specs import DatasetSpec, TrainedModel

logger = logging.getLogger(__name__)


def setup(ctx: Context) -> None:
    """Seed RNGs + drop a breadcrumb. Bookkeeping only."""
    seed = ctx.config.seed
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    logger.info("setup: run_dir=%s seed=%d", ctx.run_dir, seed)
    ctx.save_text("setup", f"slug: {ctx.config.slug()}\nseed: {seed}\n")


def prepare_dataset(ctx: Context) -> None:
    """Delegate to ctx.dataset_module.prepare(ctx) — the USER's function in
    pipeline/datasets/<name>/__init__.py."""
    # ↓ user function — every dataset module exports this (see protocols.DatasetModule).
    spec = ctx.dataset_module.prepare(ctx)
    if not isinstance(spec, DatasetSpec):
        raise TypeError(
            f"{ctx.dataset_module.__name__}.prepare(ctx) returned "
            f"{type(spec).__name__}, expected DatasetSpec"
        )
    ctx.store.put(K.DATASET, spec)
    ctx.save_json("dataset_used", spec.to_manifest())


def train(ctx: Context) -> None:
    """Delegate to ctx.model_module.train(ctx, dataset) — the USER's function
    in pipeline/models/<name>/__init__.py.

    Two functions called `train` are involved: this one (framework phase)
    vs. the user's train (the one whose contract the model tutorial documents).
    """
    ds = ctx.store.get(K.DATASET, DatasetSpec)
    # ↓ user function — every model module exports this (see protocols.ModelModule).
    trained = ctx.model_module.train(ctx, ds)
    if not isinstance(trained, TrainedModel):
        raise TypeError(
            f"{ctx.model_module.__name__}.train(ctx, dataset) returned "
            f"{type(trained).__name__}, expected TrainedModel"
        )
    ctx.store.put(K.TRAINED_MODEL, trained)


# Held-out eval set + the model's own test split. Override per-row in
# runs.yaml via `train_cfg: { eval_datasets: [...] }` if a model needs
# a different list (rare — keeping the set fixed is what makes the
# leaderboard fair).
DEFAULT_EVAL_DATASETS = ["fer2013_holdout"]


def evaluate(ctx: Context) -> None:
    """Re-load the just-trained best checkpoint and run uniform eval
    against the training dataset's test split + every dataset in
    ``ctx.config.train_cfg["eval_datasets"]`` (default: fer2013_holdout).

    Writes one bundle per dataset to ``ctx.run_dir/eval/<dataset>/``:
      summary.json + per_class.json + confusion.json + confusion.png

    Idempotent against missing pieces — a model module without
    ``build()`` logs a warning and skips, missing eval datasets log and
    skip, an empty test split records ``acc=None`` instead of crashing.
    The training run still counts as successful.
    """
    import torch

    from pipeline.eval.artifacts import write_eval_artifacts
    from pipeline.eval.loader import load_eval_dataset
    from pipeline.eval.metrics import compute_eval
    from pipeline.training.data import make_loader
    from pipeline.training.loop import auto_device

    trained = ctx.store.get(K.TRAINED_MODEL, TrainedModel)
    train_ds = ctx.store.get(K.DATASET, DatasetSpec)

    build_fn = getattr(ctx.model_module, "build", None)
    if build_fn is None:
        logger.warning(
            "evaluate: %s has no build() — skipping eval phase. The model's "
            "train() must own a re-instantiable architecture for eval to "
            "reload the checkpoint without retraining.",
            ctx.model_module.__name__,
        )
        return

    preprocess = getattr(ctx.model_module, "PREPROCESS", None)
    if preprocess is None:
        logger.warning(
            "evaluate: %s has no PREPROCESS — skipping eval. Models that "
            "want eval must export the inference-time transform.",
            ctx.model_module.__name__,
        )
        return

    device = auto_device()
    model = build_fn(train_ds.num_classes).to(device)

    if not trained.checkpoint_path.exists():
        logger.warning("evaluate: checkpoint %s missing; skipping",
                       trained.checkpoint_path)
        return
    ck = torch.load(trained.checkpoint_path, map_location=device, weights_only=False)
    # Tolerate both pipeline-saved envelopes (model_state_dict) and the
    # older notebook envelope (model_state, singular) so baselines and
    # newly-trained runs go through the same code path.
    state = ck
    if isinstance(ck, dict):
        for key in ("model_state_dict", "model_state", "state_dict"):
            if key in ck:
                state = ck[key]
                break
    model.load_state_dict(state)
    model.eval()

    # Always include the training dataset's own test split first so
    # there's an in-distribution number to anchor the OOD eval against.
    eval_names: list[str] = []
    if "test" in train_ds.splits:
        # Re-load via the dataset module so the spec we hand compute_eval
        # is the same shape as eval datasets (test-only, name match for
        # the artifact dir).
        eval_names.append(train_ds.name)
    override = ctx.config.train_cfg.get("eval_datasets")
    eval_names.extend(override if override is not None else DEFAULT_EVAL_DATASETS)

    seen: set[str] = set()
    for name in eval_names:
        if name in seen:
            continue
        seen.add(name)

        if name == train_ds.name:
            # In-distribution: use the train_ds spec we already have so
            # we don't re-import the same module.
            spec = train_ds
        else:
            try:
                spec = load_eval_dataset(name, ctx)
            except Exception:  # noqa: BLE001 — log and skip; don't fail the run
                logger.exception("evaluate: skipping %s (load failed)", name)
                continue

        test_csv = spec.splits.get("test")
        if test_csv is None:
            logger.warning("evaluate: %s has no test split; skipping", name)
            continue

        batch_size  = int(ctx.config.train_cfg.get("batch_size", 32))
        num_workers = int(ctx.config.train_cfg.get("num_workers", 0))
        loader = make_loader(test_csv, preprocess,
                             batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

        try:
            metrics = compute_eval(
                model, loader, device,
                num_classes=spec.num_classes, class_names=spec.class_names,
            )
        except Exception:  # noqa: BLE001
            logger.exception("evaluate: %s failed during compute_eval", name)
            continue

        out_dir = ctx.run_dir / "eval" / name
        write_eval_artifacts(metrics, out_dir, meta={
            "model_id":         ctx.config.slug(),
            "dataset":          name,
            "checkpoint_path":  str(trained.checkpoint_path),
            "n_samples":        metrics["n_samples"],
        })
        acc = metrics["acc"]
        logger.info("evaluate: %s — acc=%.4f macro_f1=%.4f → %s",
                    name,
                    -1.0 if acc is None else acc,
                    metrics["macro_f1"] or 0.0,
                    out_dir)
