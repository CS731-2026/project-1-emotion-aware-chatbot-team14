#!/usr/bin/env python3
"""CLI entry point for the v2 training pipeline.

Usage:
    python train.py experiments/<name>.yaml

The experiment yaml names (dataset, model, config); the driver
resolves them, builds a Context under output/run/<slug>__<ts>/, and
walks the phases listed in the yaml.

Logs go to stdout. Everything else (config snapshot, metrics jsonl,
artifacts, checkpoints) lands in the run dir.
"""

from __future__ import annotations

import argparse
import logging
import sys

from pipeline.driver import run_experiment_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one experiment.")
    parser.add_argument("experiment", help="path to experiments/<name>.yaml")
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG-level logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    ctx = run_experiment_file(args.experiment)
    print(f"\nrun dir: {ctx.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
