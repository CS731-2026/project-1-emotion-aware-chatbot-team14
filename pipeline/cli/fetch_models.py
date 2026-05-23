"""Download a Kaggle weights dataset into models/.

Pulls every file from the configured Kaggle slug into `models/<id>/`,
matching the layout `make deploy-model` produces locally. After fetch,
add the corresponding entry to `application/model_service/models.yaml`
(or run `make deploy-model` once with the same id to register it) and
set EMOTION_MODEL_ID in .env.

Usage:
    python -m pipeline.cli.fetch_models --slug team14/empathbot-checkpoints

If --slug is omitted, reads KAGGLE_WEIGHTS_SLUG from .env.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _kaggle(args: list[str]) -> int:
    try:
        return subprocess.run(["kaggle", *args]).returncode
    except FileNotFoundError:
        print("✗ kaggle CLI not found. `pip install kaggle` and set "
              "KAGGLE_USERNAME / KAGGLE_KEY in .env.", file=sys.stderr)
        return 127


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", default=os.environ.get("KAGGLE_WEIGHTS_SLUG"),
                    help="Kaggle dataset slug (default $KAGGLE_WEIGHTS_SLUG)")
    ap.add_argument("--repo-root", default=".", help="repo root (default cwd)")
    args = ap.parse_args()

    if not args.slug:
        print("✗ no slug. Pass --slug team14/empathbot-checkpoints or "
              "set KAGGLE_WEIGHTS_SLUG in .env.", file=sys.stderr)
        return 2

    repo_root = Path(args.repo_root).resolve()
    dest = repo_root / "models"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"→ kaggle datasets download -d {args.slug} -p {dest.relative_to(repo_root)} --unzip")
    rc = _kaggle(["datasets", "download", "-d", args.slug,
                   "-p", str(dest), "--unzip", "--force"])
    if rc != 0:
        print(f"✗ kaggle CLI exited {rc}", file=sys.stderr)
        return rc
    print(f"✓ fetched into {dest.relative_to(repo_root)}")
    print("  Next: register the id in application/model_service/models.yaml")
    print("  (or run `make deploy-model RUN=... ID=...` once to wire it up).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
