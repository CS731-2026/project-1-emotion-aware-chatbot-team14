"""Evaluate an existing checkpoint that wasn't produced by a recent run.

Used for the hand-trained `models/empathbot/empath_final.pth` and
`empath_best_v1.pth` weights — they predate the eval phase, so they
need a separate entry point that doesn't require a `output/run/<slug>/`
context to live in.

Usage:
    python -m pipeline.eval.baselines --id empath_final
    python -m pipeline.eval.baselines --id empath_best_v1

Output goes to ``output/eval/baseline__<id>/<dataset>/...`` — same
artifact bundle as the training-phase eval, so ``make compare`` reads
both transparently.

Datasets evaluated: every entry in --datasets (default: the empath
in-distribution test split + fer2013_holdout). Matches what the
auto-eval-after-train pass produces for a new sweep run.

Registry: BASELINES below maps each id → (checkpoint path relative to
repo root, pipeline model module). To register a new baseline, add
one row. Pulls from models.yaml are intentionally avoided — that
registry maps to *service* variants ("empathbot" / "resnet18"), but
eval needs the pipeline model module that owns build() + PREPROCESS,
which is finer-grained.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import NamedTuple


logger = logging.getLogger(__name__)


class BaselineSpec(NamedTuple):
    checkpoint: str        # path relative to repo root
    pipeline_model: str    # module name under pipeline.models.


# Hand-trained team checkpoints that predate the eval system. Add a row
# here when a new baseline lands under models/.
#
# Both current baselines were trained from Notebooks/6b_empathbot_v1_improvements.ipynb
# (the timm efficientnet_b2 variant) — their state-dict keys start with
# `encoder.conv_stem.*`, matching pipeline.models.empathbot_v1. The
# torchvision variant (pipeline.models.empathbot_final, ported from
# Notebooks/5_final_empathbot_training_v4.ipynb) has different key
# names (`backbone.0.0.*`) and won't accept these checkpoints.
BASELINES: dict[str, BaselineSpec] = {
    "empath_final": BaselineSpec(
        checkpoint="models/empathbot/empath_final.pth",
        pipeline_model="empathbot_v1",
    ),
    # empath_best_v1.pth: resnet18 backbone + manual SE; state-dict keys
    # start with `stem.0.*` — 144 keys match pipeline.models.empathbot_resnet18.
    "empath_best_v1": BaselineSpec(
        checkpoint="models/empathbot/empath_best_v1.pth",
        pipeline_model="empathbot_resnet18",
    ),
}


DEFAULT_DATASETS = ["empath", "fer2013_holdout"]


def _build_ctx_shim(repo_root: Path):
    """Lightweight stand-in for pipeline.framework.Context — provides only
    what dataset prepare() functions actually use (logger access via
    ctx.config — none of the empath/fer2013 loaders touch ctx, so we can
    pass a near-empty object). Returned object exposes a `.config` with a
    .train_cfg dict so dataset modules that probe it don't AttributeError.
    """
    class _Cfg:
        train_cfg: dict = {}
    class _Ctx:
        config = _Cfg()
        repo_root_path = repo_root
    return _Ctx()


def _load_dataset(name: str, ctx):
    from pipeline.eval.loader import load_eval_dataset
    from pipeline.framework.specs import DatasetSpec
    spec = load_eval_dataset(name, ctx)
    assert isinstance(spec, DatasetSpec)
    return spec


def _resolve_model_module(pipeline_model: str) -> ModuleType:
    return importlib.import_module(f"pipeline.models.{pipeline_model}")


# Hand-trained baselines and pipeline-trained runs use different
# checkpoint envelopes — the notebooks saved `model_state` (singular),
# the pipeline saves `model_state_dict`. Probe both before falling back
# to "assume the dict IS the state dict".
_STATE_DICT_KEYS = ("model_state_dict", "model_state", "state_dict")


def _extract_state_dict(ck):
    if not isinstance(ck, dict):
        return ck
    for key in _STATE_DICT_KEYS:
        if key in ck:
            return ck[key]
    return ck


def evaluate_baseline(
    baseline_id: str,
    *,
    repo_root: Path,
    datasets: list[str] | None = None,
) -> Path:
    """Run the full eval bundle for one baseline id. Returns the output dir."""
    import torch

    from pipeline.eval.artifacts import write_eval_artifacts
    from pipeline.eval.metrics import compute_eval
    from pipeline.training.data import make_loader
    from pipeline.training.loop import auto_device

    if baseline_id not in BASELINES:
        raise KeyError(
            f"unknown baseline id {baseline_id!r}. Known: {sorted(BASELINES)}. "
            f"Add a BASELINES entry in pipeline/eval/baselines.py."
        )
    spec = BASELINES[baseline_id]
    ckpt_path = (repo_root / spec.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"baseline checkpoint missing: {ckpt_path}\n"
            f"  This baseline expects models/{spec.checkpoint.split('/', 1)[1]} "
            f"to be on disk. Models live in the gitignored models/ tree — see "
            f"README.md → 'Switching to the real model' for where to drop it."
        )

    pipeline_model_name = spec.pipeline_model
    model_module = _resolve_model_module(pipeline_model_name)
    build_fn = getattr(model_module, "build", None)
    preprocess = getattr(model_module, "PREPROCESS", None)
    if build_fn is None or preprocess is None:
        raise RuntimeError(
            f"pipeline.models.{pipeline_model_name} must export build() + "
            f"PREPROCESS for baseline eval — got build={build_fn is not None} "
            f"preprocess={preprocess is not None}"
        )

    ctx_shim = _build_ctx_shim(repo_root)
    eval_root = repo_root / "output" / "eval" / f"baseline__{baseline_id}"
    eval_root.mkdir(parents=True, exist_ok=True)

    device = auto_device()
    requested = datasets if datasets is not None else DEFAULT_DATASETS
    summaries: list[dict] = []

    for name in requested:
        try:
            ds_spec = _load_dataset(name, ctx_shim)
        except Exception:  # noqa: BLE001 — log and skip
            logger.exception("baseline %s: skipping dataset %s (load failed)",
                             baseline_id, name)
            continue

        test_csv = ds_spec.splits.get("test")
        if test_csv is None:
            logger.warning("baseline %s: %s has no test split; skipping",
                           baseline_id, name)
            continue

        # Build architecture sized to this dataset's class count, load
        # weights once per dataset (cheap — checkpoint already in RAM
        # via torch.load below if we wanted to optimise, but the simple
        # path is easier to read).
        model = build_fn(ds_spec.num_classes).to(device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = _extract_state_dict(ck)
        model.load_state_dict(state)
        model.eval()

        loader = make_loader(test_csv, preprocess,
                             batch_size=32, shuffle=False, num_workers=0)
        metrics = compute_eval(
            model, loader, device,
            num_classes=ds_spec.num_classes, class_names=ds_spec.class_names,
        )

        out_dir = eval_root / name
        write_eval_artifacts(metrics, out_dir, meta={
            "model_id":         f"baseline__{baseline_id}",
            "dataset":          name,
            "checkpoint_path":  str(ckpt_path),
            "pipeline_model":   pipeline_model_name,
            "evaluated_at":     datetime.now().isoformat(timespec="seconds"),
        })
        logger.info("baseline %s × %s — acc=%.4f macro_f1=%.4f → %s",
                    baseline_id, name,
                    -1.0 if metrics["acc"] is None else metrics["acc"],
                    metrics["macro_f1"] or 0.0,
                    out_dir)
        summaries.append({
            "dataset": name,
            "acc":      metrics["acc"],
            "macro_f1": metrics["macro_f1"],
        })

    # Top-level index file the compare CLI can find without scanning
    # subdirs first.
    (eval_root / "index.json").write_text(json.dumps({
        "baseline_id":     baseline_id,
        "checkpoint_path": str(ckpt_path),
        "pipeline_model":  pipeline_model_name,
        "evaluated_at":    datetime.now().isoformat(timespec="seconds"),
        "datasets":        summaries,
    }, indent=2))
    return eval_root


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", required=True, help=f"baseline id; one of {sorted(BASELINES)}")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help=f"override eval dataset list (default: {DEFAULT_DATASETS})")
    ap.add_argument("--repo-root", default=".", help="repo root (default cwd)")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    try:
        out = evaluate_baseline(args.id, repo_root=repo_root,
                                datasets=args.datasets)
    except (KeyError, FileNotFoundError, RuntimeError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2
    print(f"✓ baseline {args.id} → {out.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
