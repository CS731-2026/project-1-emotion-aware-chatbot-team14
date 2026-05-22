"""Entry point. Imports + registers what's in the sweep, runs it.

    python -m pipeline.train

The three lists below are the single answer to "what runs?". Adding a
model / dataset / config to the sweep is one new file in the right
folder + one line in the corresponding list here. Staging a WIP file
without including it: drop the file, don't add the line.
"""

from __future__ import annotations

import argparse
import logging
import sys

import configs.baseline as baseline_cfg
import configs.fast as fast_cfg
import configs.thorough as thorough_cfg
import datasets.fer2013 as fer2013
import datasets.synthetic_imbalanced as synthetic_imbalanced
import datasets.synthetic_smoke as synthetic_smoke

from models import mlp, resnet18, tiny_cnn

from pipeline.driver import sweep


# ---- registration: edit here to change what runs --------------------------

MODELS   = [mlp, tiny_cnn, resnet18]
DATASETS = [synthetic_smoke, synthetic_imbalanced, fer2013]
CONFIGS  = [fast_cfg, baseline_cfg, thorough_cfg]

# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep dataset × model × config.")
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG-level logs")
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="stop on first failure (default: log + continue across cells)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    contexts = sweep(
        datasets=DATASETS,
        models=MODELS,
        configs=CONFIGS,
        fail_fast=args.fail_fast,
    )
    print(f"\nsweep complete: {len(contexts)} run(s)")
    for c in contexts:
        print(f"  {c.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
