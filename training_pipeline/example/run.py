"""Entry point for the example classifier pipeline.

Run from the repo root:
    python example/run.py --routine train_classifier \
        --config example/base.yaml \
        --config example/debug.yaml

In a real project, `harness` would be an installed package (pip install -e .).
For this repo, we add the repo root to sys.path so imports resolve without
installation.
"""
import sys
from pathlib import Path

# Repo root → makes `import harness` resolve during development
sys.path.insert(0, str(Path(__file__).parent.parent))

import harness
from pipeline import (
    build_model,
    evaluate,
    load_data,
    save_best_checkpoint,
    save_summary,
    setup_device,
    setup_outputs,
    setup_optimizer,
    train_epoch,
    validate,
)

_RUNS_DIR = Path(__file__).parent / "runs"


def train_classifier(config: dict) -> None:
    harness.run(
        steps=[
            setup_device,
            setup_outputs,
            load_data,
            build_model,
            setup_optimizer,
            harness.loop(
                train_epoch, validate, save_best_checkpoint,
                n=lambda config: config["epochs"],
                iter_name="epoch",
            ),
            evaluate,
            save_summary,
        ],
        name="train_classifier",
        config=config,
        runs_dir=_RUNS_DIR,
    )


if __name__ == "__main__":
    harness.main(routines={"train_classifier": train_classifier})
