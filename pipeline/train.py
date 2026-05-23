"""Entry point. Declares the runs to execute, then runs them.

    python -m pipeline.train

The RUNS list below is the single answer to "what runs?". Each entry is
one (dataset, model, config) triple. Not every model belongs with every
dataset / config — declare the specific pairings worth training.

Adding a run = one new file in datasets/ or models/ or configs/ (if
needed) + one line in RUNS. Removing a run = delete or comment out
its line.
"""

from __future__ import annotations

import argparse
import logging
import sys

import configs.baseline as baseline_cfg
import configs.fast as fast_cfg
import configs.thorough as thorough_cfg

from pipeline.datasets import (
    empath,
    fer2013,
    kash,
    synthetic_imbalanced,
    synthetic_smoke,
)
from pipeline.driver import sweep
from pipeline.models import (
    ada_df,
    empathbot_final,
    empathbot_v1,
    mlp,
    posterplus,
    resnet18,
    tiny_cnn,
)


# ---- what to run ---------------------------------------------------------
# (dataset, model, config) triples. One line = one training run.

RUNS = [
    # ─── Smoke tests ─────────────────────────────────────────────────────
    # Fast, network-free (synthetic data), prove the pipeline + each
    # architecture wires up end-to-end. Heavy models (EfficientNet-B2 +
    # pretrained weights) still run in seconds at fast config (1 epoch).
    (synthetic_smoke,       mlp,              fast_cfg),
    (synthetic_smoke,       tiny_cnn,         fast_cfg),
    (synthetic_smoke,       resnet18,         fast_cfg),
    (synthetic_smoke,       ada_df,           fast_cfg),
    (synthetic_smoke,       empathbot_v1,     fast_cfg),
    (synthetic_smoke,       empathbot_final,  fast_cfg),

    # ─── Class-imbalance exercise ────────────────────────────────────────
    # Validates class_weights: auto without needing Kaggle.
    (synthetic_imbalanced,  tiny_cnn,         baseline_cfg),

    # ─── Real training on real data (requires Kaggle creds) ──────────────
    # Comment out individual lines (or the whole block) if Kaggle isn't
    # set up locally — the smoke + imbalance runs above still cover the
    # pipeline plumbing.
    (fer2013,               tiny_cnn,         baseline_cfg),
    (fer2013,               resnet18,         thorough_cfg),
    (fer2013,               ada_df,           thorough_cfg),
    (fer2013,               empathbot_v1,     thorough_cfg),
    (fer2013,               empathbot_final,  thorough_cfg),

    # POSTER++ requires the POSTER_V2 repo cloned via `make init` (the
    # .thread file in pipeline/models/posterplus/ pulls it in).
    # Uncomment after running make init.
    # (fer2013,               posterplus,       thorough_cfg),

    # ─── Local-disk datasets (uncomment after pointing env vars at sources) ──
    # kash needs $KASH_DATASET_DIR or output/data/kash/raw/ populated
    # (see Notebooks/7_kash_dataset_prep.ipynb for the expected layout).
    # (kash,                  tiny_cnn,         baseline_cfg),
    # (kash,                  empathbot_v1,     thorough_cfg),

    # empath needs at least one of EMPATH_{AFFECTNET,RAFDB,SFEW}_DIR set
    # (see Notebooks/1_dataset_pipeline.ipynb).
    # (empath,                tiny_cnn,         baseline_cfg),
    # (empath,                empathbot_v1,     thorough_cfg),
    # (empath,                empathbot_final,  thorough_cfg),
]

# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the declared RUNS list.")
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG-level logs")
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="stop on first failure (default: log + continue across runs)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    contexts = sweep(RUNS, fail_fast=args.fail_fast)
    print(f"\nsweep complete: {len(contexts)} run(s)")
    for c in contexts:
        print(f"  {c.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
