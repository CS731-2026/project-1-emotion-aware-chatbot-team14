"""Resolve a registry-mapped checkpoint path, fetching from Kaggle if missing.

The model service's factory looks up `EMOTION_MODEL_ID` in models.yaml,
gets back a relative path under `models/`, and asks this module to
return the absolute path. If the file isn't on disk, we silently
attempt to pull the team's Kaggle weights dataset before raising.

The "creds-only" team flow:
    1. teammate sets KAGGLE_USERNAME / KAGGLE_KEY in .env
    2. teammate sets EMOTION_MODEL_ID=<id> in .env
    3. teammate runs `make dev`
    4. service finds models/<id>/best.pth missing, runs kaggle datasets
       download under the hood, then loads the model
    5. subsequent boots skip the fetch (file is cached locally)

If KAGGLE_USERNAME is unset, we skip the fetch and surface the
original FileNotFoundError so the developer knows what's wrong.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


DEFAULT_SLUG = "team14/empathbot-checkpoints"


def _kaggle_creds_present() -> bool:
    """Either env vars or ~/.kaggle/kaggle.json must exist for the CLI."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    return cfg.exists()


def _fetch_kaggle_weights(models_root: Path, slug: str) -> bool:
    """Run `kaggle datasets download --unzip` into models/. Returns True
    on success, False (with a logged warning) on any failure — the
    caller decides whether to raise."""
    models_root.mkdir(parents=True, exist_ok=True)
    logger.info("model_resolver: fetching %s → %s", slug, models_root)
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug,
             "-p", str(models_root), "--unzip", "--force"],
            capture_output=True, text=True, timeout=300,
        )
    except FileNotFoundError:
        logger.warning("model_resolver: kaggle CLI not installed; skipping fetch")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("model_resolver: kaggle fetch timed out after 5 min")
        return False

    if result.returncode != 0:
        logger.warning("model_resolver: kaggle fetch exited %d. stderr: %s",
                       result.returncode, result.stderr.strip()[:500])
        return False
    return True


def resolve_checkpoint(rel_path: str, *, repo_root: Path,
                        model_id: str | None = None) -> Path:
    """Return absolute path to a checkpoint, fetching it from the team
    Kaggle weights dataset if missing locally.

    Args:
      rel_path:  path stored in models.yaml (relative to repo_root).
      repo_root: absolute path to the repo root.
      model_id:  the registry id, only used in error messages.

    Raises FileNotFoundError if the file isn't on disk after the fetch
    attempt — message names the slug + suggests `make fetch-models` so
    the developer has a manual path.
    """
    abs_path = (repo_root / rel_path).resolve()
    if abs_path.exists():
        return abs_path

    slug = os.environ.get("KAGGLE_WEIGHTS_SLUG", DEFAULT_SLUG)

    if not _kaggle_creds_present():
        raise FileNotFoundError(
            f"checkpoint missing: {abs_path}\n"
            f"  EMOTION_MODEL_ID points at this path via models.yaml "
            f"({model_id or '?'}), but the file isn't on disk and no Kaggle "
            f"creds are set (need KAGGLE_USERNAME/KAGGLE_KEY in .env, "
            f"or ~/.kaggle/kaggle.json). Either drop the .pth at that path "
            f"manually or add creds + retry."
        )

    fetched = _fetch_kaggle_weights(repo_root / "models", slug)
    if fetched and abs_path.exists():
        logger.info("model_resolver: ✓ fetched from Kaggle, using %s", abs_path)
        return abs_path

    raise FileNotFoundError(
        f"checkpoint missing: {abs_path}\n"
        f"  Tried to auto-fetch from Kaggle dataset {slug} but the file "
        f"still isn't on disk. Possible causes:\n"
        f"    - the dataset doesn't have a {model_id or rel_path} subdir\n"
        f"    - Kaggle creds are wrong or the dataset is private\n"
        f"    - the file was deleted from a previous publish-models\n"
        f"  Try `make fetch-models` for verbose output, or publish the "
        f"model first via `make publish-models`."
    )
