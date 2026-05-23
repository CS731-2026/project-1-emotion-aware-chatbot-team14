"""Publish every model under models/ to one Kaggle dataset version.

Uploads `models/*/` (subdir per model id) as a single Kaggle dataset
so a teammate's `make fetch-models` pulls every model the team has
deployed. The dataset version always reflects the team's full
canonical weights — no "I published v1 but you uploaded v2 and now
my model's gone" surprises.

Usage:
    python -m pipeline.cli.publish_models --message "v4 final + v3 sweep"
    python -m pipeline.cli.publish_models --new-dataset   # first-time only

Auth via KAGGLE_USERNAME / KAGGLE_KEY in .env or ~/.kaggle/kaggle.json
(same as fer2013 download). Slug from KAGGLE_WEIGHTS_SLUG env var,
else falls back to a sensible default.

The legacy `--id <id>` flag still works but is discouraged — it
publishes just one model and clobbers the dataset to contain only
that, deleting every other team weight from Kaggle. Use the default
(no --id) instead.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_SLUG = os.environ.get("KAGGLE_WEIGHTS_SLUG", "team14/empathbot-checkpoints")


def _write_metadata(dest: Path, slug: str, title: str) -> None:
    """Kaggle CLI requires a dataset-metadata.json at the upload root."""
    meta = {
        "title": title,
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (dest / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


def _stage(models_root: Path, work_dir: Path, slug: str, title: str,
            single_id: str | None) -> tuple[Path, list[str]]:
    """Mirror models/<id>/ (all of them, or just `single_id`) into a
    clean upload dir alongside the metadata. Returns (dir, ids_included)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    for child in work_dir.iterdir():
        if child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)

    ids: list[str] = []
    for entry in sorted(models_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(("models--", ".")):  # HF cache, hidden
            continue
        if single_id and entry.name != single_id:
            continue
        dst = work_dir / entry.name
        shutil.copytree(entry, dst)
        ids.append(entry.name)

    _write_metadata(work_dir, slug, title)
    return work_dir, ids


def _kaggle(args: list[str]) -> int:
    try:
        return subprocess.run(["kaggle", *args]).returncode
    except FileNotFoundError:
        print("✗ kaggle CLI not found. `pip install kaggle` and set "
              "KAGGLE_USERNAME / KAGGLE_KEY in .env.", file=sys.stderr)
        return 127


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", default=DEFAULT_SLUG,
                    help="Kaggle dataset slug (default $KAGGLE_WEIGHTS_SLUG "
                         "or 'team14/empathbot-checkpoints')")
    ap.add_argument("--message", default="auto-published from pipeline",
                    help="version note (only used when not --new-dataset)")
    ap.add_argument("--new-dataset", action="store_true",
                    help="first-time publish: kaggle datasets create instead of version")
    ap.add_argument("--id", default=None,
                    help="DISCOURAGED — publish only one model id, clobbering "
                         "every other team weight currently on Kaggle. Use the "
                         "default (no --id) to publish all local models.")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    models_root = repo_root / "models"
    if not models_root.is_dir():
        print(f"✗ {models_root} not found. Run `make deploy-model` first.",
              file=sys.stderr)
        return 2

    work_dir = repo_root / "output" / "kaggle_upload"
    title = f"empathbot-checkpoints — {args.id}" if args.id \
            else "empathbot-checkpoints — team weights"
    staged, ids = _stage(models_root, work_dir, args.slug, title, args.id)
    if not ids:
        scope = f"id={args.id!r}" if args.id else "any model id"
        print(f"✗ no model subdirs under {models_root} match {scope}.",
              file=sys.stderr)
        return 2
    print(f"→ staged {len(ids)} model(s) at {staged.relative_to(repo_root)}: {ids}")

    if args.id and not args.new_dataset:
        print(f"⚠  --id {args.id!r} publishes ONLY that model and removes "
              f"every other model from the Kaggle dataset. Re-run without "
              f"--id to publish all local models.", file=sys.stderr)

    if args.new_dataset:
        rc = _kaggle(["datasets", "create", "-p", str(staged), "--dir-mode", "zip"])
    else:
        rc = _kaggle(["datasets", "version", "-p", str(staged),
                       "-m", args.message, "--dir-mode", "zip"])

    if rc != 0:
        print(f"✗ kaggle CLI exited {rc}", file=sys.stderr)
        return rc
    print(f"✓ {len(ids)} model(s) published to {args.slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
