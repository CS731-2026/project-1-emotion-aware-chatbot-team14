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

from pipeline.datasets import fer2013, synthetic_imbalanced, synthetic_smoke
from pipeline.driver import sweep
from pipeline.models import mlp, resnet18, tiny_cnn


# ---- what to run ---------------------------------------------------------
# (dataset, model, config) triples. One line = one training run.

RUNS = [
    # Smoke tests — fast, network-free, prove the pipeline wires up.
    (synthetic_smoke,       mlp,        fast_cfg),
    (synthetic_smoke,       tiny_cnn,   fast_cfg),

    # Imbalanced data — exercise class_weights: auto without Kaggle.
    (synthetic_imbalanced,  tiny_cnn,   baseline_cfg),

    # Real training on real data. Comment out if Kaggle creds aren't set up.
    (fer2013,               tiny_cnn,   baseline_cfg),
    (fer2013,               resnet18,   thorough_cfg),
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
