"""Kaggle utility surface, one import, every Kaggle-touching operation.

Wraps the `kaggle` CLI so notebooks, dataset modules, and ad-hoc
scripts can call into Kaggle without each one re-implementing the
subprocess plumbing. Auth flows through the standard mechanisms:

  - KAGGLE_USERNAME / KAGGLE_KEY in .env (read at startup)
  - ~/.kaggle/kaggle.json (fallback the CLI checks itself)

Functions:

  download_dataset(slug, dest)         pull a Kaggle dataset to a local dir
  fetch_models(dest=models/, slug=…)   pull team weights dataset
  publish_models(models_dir, …)        push every models/<id>/ to one Kaggle version
  dataset_exists(slug)                 quick check before publish/create
  creds_present()                      bool, used to gate auto-fetch attempts

Most teammates don't need this directly, `make fetch-models`,
`make publish-models`, and the model service's auto-fetch all wrap
these. Import directly only if you need Kaggle ops from a notebook
or a custom dataset module.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


DEFAULT_WEIGHTS_SLUG = os.environ.get(
    "KAGGLE_WEIGHTS_SLUG", "team14/empathbot-checkpoints",
)


# ---- credential check ----------------------------------------------------

def creds_present() -> bool:
    """True iff either env vars or ~/.kaggle/kaggle.json gives the CLI
    something to authenticate with."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


# ---- raw CLI wrapper -----------------------------------------------------

class KaggleCLIError(RuntimeError):
    """Raised when the kaggle CLI exits non-zero (or isn't installed)."""


def _kaggle(*args: str, timeout: int = 300, check: bool = True) -> str:
    """Run `kaggle <args>` and return stdout. Raises KaggleCLIError on
    failure (when check=True)."""
    try:
        result = subprocess.run(
            ["kaggle", *args], capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError as e:
        raise KaggleCLIError(
            "kaggle CLI not found. `pip install kaggle` and set "
            "KAGGLE_USERNAME / KAGGLE_KEY in .env."
        ) from e
    if check and result.returncode != 0:
        raise KaggleCLIError(
            f"kaggle {' '.join(args)} exited {result.returncode}.\n"
            f"  stdout: {result.stdout.strip()[:400]}\n"
            f"  stderr: {result.stderr.strip()[:400]}"
        )
    return result.stdout


# ---- datasets ------------------------------------------------------------

def download_dataset(slug: str, dest: Path | str, *,
                      unzip: bool = True, force: bool = True) -> Path:
    """Pull a Kaggle dataset (e.g. 'msambare/fer2013') into `dest`.

    Mirror of `make fetch-models` for the generic case. Used by
    pipeline/datasets/fer2013 to bring its source down on first prep.
    Returns the destination Path.
    """
    dest_p = Path(dest)
    dest_p.mkdir(parents=True, exist_ok=True)
    if not creds_present():
        raise KaggleCLIError(
            "no Kaggle creds. Set KAGGLE_USERNAME / KAGGLE_KEY in .env "
            "(or drop ~/.kaggle/kaggle.json)."
        )
    args = ["datasets", "download", "-d", slug, "-p", str(dest_p)]
    if unzip:
        args.append("--unzip")
    if force:
        args.append("--force")
    logger.info("kaggle: downloading %s → %s", slug, dest_p)
    _kaggle(*args)
    return dest_p


def dataset_exists(slug: str) -> bool:
    """True if `slug` resolves to a published Kaggle dataset."""
    try:
        out = _kaggle("datasets", "list", "-s", slug.split("/", 1)[-1], check=False)
    except KaggleCLIError:
        return False
    return slug in out


def fetch_models(dest: Path | str = "models",
                  slug: str = DEFAULT_WEIGHTS_SLUG) -> Path:
    """Convenience wrapper, pull the team weights dataset into models/.
    Equivalent to `make fetch-models`."""
    return download_dataset(slug, dest, unzip=True, force=True)


# ---- model publication ---------------------------------------------------

def _write_metadata(stage_dir: Path, slug: str, title: str) -> None:
    """Kaggle CLI looks for dataset-metadata.json at the upload root."""
    (stage_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": title,
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))


def publish_models(models_dir: Path | str = "models", *,
                    stage_dir: Path | str = "output/kaggle_upload",
                    slug: str = DEFAULT_WEIGHTS_SLUG,
                    message: str = "auto-published from pipeline.kaggle",
                    new_dataset: bool = False,
                    include_only: list[str] | None = None) -> list[str]:
    """Upload every `models/<id>/` subdir as one Kaggle dataset version.

    Mirror of `make publish-models`. Returns the list of model ids
    actually included.

    `include_only=[...]` restricts to specific ids (rarely useful since
    Kaggle datasets are atomic, see the script docstring for why
    publishing one model at a time is discouraged).
    """
    if not creds_present():
        raise KaggleCLIError(
            "no Kaggle creds. Set KAGGLE_USERNAME / KAGGLE_KEY in .env "
            "(or drop ~/.kaggle/kaggle.json)."
        )

    src = Path(models_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"{src} not found, nothing to publish")

    stage = Path(stage_dir)
    stage.mkdir(parents=True, exist_ok=True)
    for child in stage.iterdir():
        if child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)

    ids: list[str] = []
    for entry in sorted(src.iterdir()):
        if not entry.is_dir() or entry.name.startswith(("models--", ".")):
            continue
        if include_only and entry.name not in include_only:
            continue
        shutil.copytree(entry, stage / entry.name)
        ids.append(entry.name)

    if not ids:
        raise FileNotFoundError(f"no model subdirs under {src} matched")

    title = f"empathbot-checkpoints, {len(ids)} model(s)"
    _write_metadata(stage, slug, title)
    logger.info("kaggle: %s %d model(s) → %s",
                "creating" if new_dataset else "versioning", len(ids), slug)

    if new_dataset:
        _kaggle("datasets", "create", "-p", str(stage), "--dir-mode", "zip")
    else:
        _kaggle("datasets", "version", "-p", str(stage),
                 "-m", message, "--dir-mode", "zip")
    return ids
