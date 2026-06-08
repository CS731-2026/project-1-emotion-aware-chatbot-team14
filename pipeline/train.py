"""Entry point. Loads runs.yaml and executes every enabled entry.

    python -m pipeline.train                    # reads ./runs.yaml
    python -m pipeline.train --runs other.yaml  # custom file
    python -m pipeline.train --fail-fast        # stop on first failure
    python -m pipeline.train -v                 # DEBUG-level logs

runs.yaml is the single answer to "what runs?". Each entry names a
dataset, model, and config, plus an optional `train_cfg:` block for
per-run hyperparameter overrides. See pipeline/MIGRATING_NOTEBOOKS.md
for the schema.
"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

# Load .env from cwd before any module touches os.environ, needed so
# datasets/fer2013 picks up KAGGLE_USERNAME / KAGGLE_KEY (kaggle CLI
# checks env vars before ~/.kaggle/kaggle.json), and so the EMPATH_* /
# KASH_* opt-in flags can be set per-run from .env.
load_dotenv()

from pipeline.driver import sweep
from pipeline.runs_loader import load_runs


def _filter_runs(resolved: list, pattern: str) -> list:
    """Keep runs whose slug contains `pattern`. Match is case-insensitive
    and checks the synthesised '<dataset> <model> <config>' string so
    `--run my_model` or `--run fer2013` or `--run thorough` all work."""
    needle = pattern.lower()
    out = []
    for r in resolved:
        slug = f"{r.dataset.NAME} {r.model.__name__.rsplit('.', 1)[-1]} {r.config.NAME}".lower()
        if needle in slug:
            out.append(r)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every enabled entry in a runs.yaml file.")
    parser.add_argument("--runs", default="runs.yaml",
                        help="path to runs file (default: ./runs.yaml)")
    parser.add_argument("--run", default=None,
                        help="case-insensitive substring filter, run only entries "
                             "whose dataset/model/config slug contains this string")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="DEBUG-level logs")
    parser.add_argument("--fail-fast", action="store_true",
                        help="stop on first failure (default: log + continue across runs)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    resolved = load_runs(args.runs)
    if not resolved:
        print(f"no enabled runs in {args.runs}", file=sys.stderr)
        return 1

    if args.run:
        before = len(resolved)
        resolved = _filter_runs(resolved, args.run)
        if not resolved:
            print(f"no runs match --run {args.run!r} ({before} entries in {args.runs})",
                  file=sys.stderr)
            return 1
        print(f"--run {args.run!r}: matched {len(resolved)} of {before} run(s)")

    triples = [r.as_tuple() for r in resolved]
    contexts = sweep(triples, fail_fast=args.fail_fast)
    print(f"\nsweep complete: {len(contexts)} run(s)")
    for c in contexts:
        print(f"  {c.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
