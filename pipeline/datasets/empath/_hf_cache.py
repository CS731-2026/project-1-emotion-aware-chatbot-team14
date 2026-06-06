"""Purge the HuggingFace dataset cache for a specific slug.

After ``affectnet._download_to()`` / ``rafdb._download_to()`` materialize
the face-cropped JPEGs to ``output/data/empath/raw/<source>/``, the
parquet copies under ``~/.cache/huggingface/`` are pure redundancy —
the loaders never read from the HF cache again (the on-disk JPEGs
satisfy the cache check). For large datasets like AffectNet-HQ that
~6 GB of parquet is silent disk bloat.

Call ``purge_hf_dataset_cache("Piro17/affectnethq")`` at the end of a
successful ``_download_to`` to release it. Set the env var
``EMPATH_KEEP_HF_CACHE=1`` to keep the cache (useful if you're
iterating on the materialization logic and don't want to re-download).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def purge_hf_dataset_cache(slug: str) -> None:
    """Remove the HF hub + datasets cache for a specific dataset slug.

    Slug format: ``"<org>/<dataset>"`` (e.g. ``"Piro17/affectnethq"``).
    HF stores it under ``~/.cache/huggingface/hub/datasets--<org>--<dataset>/``
    and a parallel arrow dataset under ``~/.cache/huggingface/datasets/<org>___<dataset>/``.

    Idempotent — missing dirs are silently skipped. Errors logged but
    not raised; cache purge shouldn't fail a successful materialization.
    """
    if os.environ.get("EMPATH_KEEP_HF_CACHE", "0") == "1":
        logger.info("hf_cache: EMPATH_KEEP_HF_CACHE=1 — leaving %s alone", slug)
        return

    home = Path.home() / ".cache" / "huggingface"
    org, name = slug.split("/", 1)
    candidates = [
        home / "hub" / f"datasets--{org}--{name}",
        home / "datasets" / f"{org}___{name}",
    ]
    freed = 0
    for path in candidates:
        if not path.exists():
            continue
        try:
            size = _dir_size(path)
            shutil.rmtree(path)
            freed += size
        except OSError as e:
            logger.warning("hf_cache: failed to purge %s: %s", path, e)
    if freed:
        logger.info("hf_cache: purged %.1f MB for %s", freed / 1024 / 1024, slug)


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total
