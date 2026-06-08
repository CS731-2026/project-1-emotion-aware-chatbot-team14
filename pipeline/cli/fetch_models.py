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
import shutil
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
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing models/<id>/ subdirs that conflict "
                         "with the downloaded dataset. Without this flag, fetch "
                         "downloads to a staging dir and refuses to clobber any "
                         "id that's already on disk, your local checkpoints "
                         "(e.g. models/empathbot/empath_final.pth) stay safe.")
    args = ap.parse_args()

    if not args.slug:
        print("✗ no slug. Pass --slug team14/empathbot-checkpoints or "
              "set KAGGLE_WEIGHTS_SLUG in .env.", file=sys.stderr)
        return 2

    repo_root = Path(args.repo_root).resolve()
    models_root = repo_root / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    # Stage first into output/kaggle_fetch/, then conflict-check before
    # touching models/. Prevents the kaggle CLI's --force unzip from
    # silently overwriting hand-trained checkpoints already on disk.
    staging = repo_root / "output" / "kaggle_fetch"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print(f"→ kaggle datasets download -d {args.slug} -p "
          f"{staging.relative_to(repo_root)} --unzip")
    rc = _kaggle(["datasets", "download", "-d", args.slug,
                   "-p", str(staging), "--unzip", "--force"])
    if rc != 0:
        print(f"✗ kaggle CLI exited {rc}", file=sys.stderr)
        return rc

    fetched_ids = [p.name for p in sorted(staging.iterdir())
                    if p.is_dir() and not p.name.startswith((".", "models--"))]
    if not fetched_ids:
        print(f"✗ no model subdirs found in {staging.relative_to(repo_root)} "
              f"after download.", file=sys.stderr)
        return 2

    conflicts = [i for i in fetched_ids if (models_root / i).exists()]
    if conflicts and not args.force:
        print(f"✗ refusing to overwrite existing model dirs in "
              f"{models_root.relative_to(repo_root)}:", file=sys.stderr)
        for i in conflicts:
            print(f"    models/{i}/", file=sys.stderr)
        print(f"  Downloaded copy is staged at "
              f"{staging.relative_to(repo_root)}/, move what you want by "
              f"hand, or re-run with --force to overwrite.", file=sys.stderr)
        return 2

    moved: list[str] = []
    skipped: list[str] = []
    for i in fetched_ids:
        src = staging / i
        dst = models_root / i
        if dst.exists():
            shutil.rmtree(dst)  # --force path only, guarded above
        shutil.move(str(src), str(dst))
        moved.append(i)
    print(f"✓ fetched {len(moved)} model(s) into "
          f"{models_root.relative_to(repo_root)}/: {moved}")
    if skipped:
        print(f"  skipped (already on disk): {skipped}")
    print("  Next: register the id in application/model_service/models.yaml")
    print("  (or run `make deploy-model RUN=... ID=...` once to wire it up).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
