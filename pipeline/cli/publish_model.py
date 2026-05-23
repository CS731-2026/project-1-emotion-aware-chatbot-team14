"""Publish a deployed model checkpoint to a Kaggle dataset.

Kaggle's "datasets" feature accepts any blob — the team uses it as a
free, versioned store for trained checkpoints so we never have to
commit `.pth` binaries to git. One Kaggle dataset slug holds every
deployed model id; each `publish` call is a new version.

Usage:
    python -m pipeline.cli.publish_model --id empathbot_final --message "v4 final"
    python -m pipeline.cli.publish_model --id empathbot_final --new-dataset

Auth via KAGGLE_USERNAME / KAGGLE_KEY in .env or ~/.kaggle/kaggle.json
(same as fer2013 download). The default slug is read from
KAGGLE_WEIGHTS_SLUG env var, else falls back to a placeholder.
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
    """Kaggle CLI requires a dataset-metadata.json at the upload root.
    See https://github.com/Kaggle/kaggle-api#create-a-new-dataset."""
    meta = {
        "title": title,
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (dest / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


def _stage_for_upload(model_dir: Path, work_dir: Path, slug: str, title: str) -> Path:
    """Copy the model files into a clean upload dir alongside the metadata."""
    work_dir.mkdir(parents=True, exist_ok=True)
    for child in work_dir.iterdir():
        if child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)
    for src in model_dir.iterdir():
        if src.is_file():
            shutil.copy2(src, work_dir / src.name)
    _write_metadata(work_dir, slug, title)
    return work_dir


def _kaggle(args: list[str]) -> int:
    """Run a kaggle CLI command. Returns its exit code."""
    try:
        return subprocess.run(["kaggle", *args]).returncode
    except FileNotFoundError:
        print("✗ kaggle CLI not found. `pip install kaggle` and set "
              "KAGGLE_USERNAME / KAGGLE_KEY in .env.", file=sys.stderr)
        return 127


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", required=True,
                    help="model id under models/<id>/ to publish")
    ap.add_argument("--slug", default=DEFAULT_SLUG,
                    help="Kaggle dataset slug (default $KAGGLE_WEIGHTS_SLUG "
                         "or 'team14/empathbot-checkpoints')")
    ap.add_argument("--message", default="auto-published from pipeline",
                    help="version note (only used when not --new-dataset)")
    ap.add_argument("--new-dataset", action="store_true",
                    help="first-time publish: kaggle datasets create instead of version")
    ap.add_argument("--repo-root", default=".", help="repo root (default cwd)")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    model_dir = repo_root / "models" / args.id
    if not model_dir.is_dir():
        print(f"✗ models/{args.id}/ not found. Run `make deploy-model` first.",
              file=sys.stderr)
        return 2
    if not any(model_dir.glob("*.pth")):
        print(f"✗ no .pth file under {model_dir}", file=sys.stderr)
        return 2

    work_dir = repo_root / "output" / "kaggle_upload" / args.id
    title = f"empathbot-checkpoints — {args.id}"
    staged = _stage_for_upload(model_dir, work_dir, args.slug, title)
    print(f"→ staged for upload at {staged.relative_to(repo_root)}")

    if args.new_dataset:
        print(f"→ kaggle datasets create -p {staged}")
        rc = _kaggle(["datasets", "create", "-p", str(staged), "--dir-mode", "zip"])
    else:
        print(f"→ kaggle datasets version -p {staged} -m {args.message!r}")
        rc = _kaggle(["datasets", "version", "-p", str(staged),
                       "-m", args.message, "--dir-mode", "zip"])

    if rc != 0:
        print(f"✗ kaggle CLI exited {rc}", file=sys.stderr)
        return rc
    print(f"✓ {args.id} published to {args.slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
