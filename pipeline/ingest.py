"""Dataset ingest helpers — split out of phases.py so the phase
function stays a thin orchestrator.

Each helper does one thing and is independently testable:
  - download_kaggle:  shell out to the kaggle CLI, into a cache dir
  - md5_of_dir:       hash the contents of an extracted source so we can
                      skip re-prep when nothing changed
  - scan_imagefolder: walk train_dir/<class>/*.{png,jpg} → (path, src_label)
  - apply_remap:      source label string → target class index (or drop)
  - carve_val:        random split a train df into train / val by seed
  - compute_class_weights: inverse-frequency vector for a label column
  - write_split_csvs: persist train.csv / val.csv / test.csv

Network call is isolated to download_kaggle; everything else is pure
pandas / pathlib and easy to unit test with a fake dir.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_IMG_EXTS = {".png", ".jpg", ".jpeg"}
DROP_SENTINEL = "__drop__"


def generate_synthetic(
    dest: Path,
    *,
    class_names: list[str],
    samples_per_class: dict[str, int] | int,
    seed: int = 42,
    image_size: int = 32,
) -> Path:
    """Build a fake imagefolder dataset at `dest` so the pipeline can
    smoke-test without network or real data.

    Each class gets `samples_per_class[name]` (or a flat int) images,
    drawn from a class-specific mean + noise so a model can actually
    learn something meaningful — the smoke run validates "training
    drives loss down" not just "the wiring works".

    Idempotent: if `dest` already has subdirs, returns immediately —
    the per-dataset md5 cache in prepare_dataset still picks up changes
    in image content if a teammate edits a generated file by hand.
    """
    import numpy as np
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    if any(p.is_dir() for p in dest.iterdir()):
        logger.info("synthetic: cache present at %s; skipping generation", dest)
        return dest

    rng = np.random.default_rng(seed)
    # Per-class mean color so the classification problem is learnable.
    class_means = rng.integers(40, 215, size=(len(class_names), 3), endpoint=True)

    if isinstance(samples_per_class, int):
        samples_per_class = {n: samples_per_class for n in class_names}

    logger.info("synthetic: generating %s in %s",
                {n: samples_per_class[n] for n in class_names}, dest)
    for class_idx, name in enumerate(class_names):
        class_dir = dest / name
        class_dir.mkdir()
        mean = class_means[class_idx].astype(np.float32)
        for i in range(samples_per_class[name]):
            noise = rng.normal(0, 35, size=(image_size, image_size, 3))
            img = np.clip(mean + noise, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(class_dir / f"{i:04d}.png")
    return dest


def download_kaggle(dataset_id: str, dest: Path) -> Path:
    """Download a Kaggle dataset archive into `dest` and extract it in place.

    Uses the kaggle CLI (auth via ~/.kaggle/kaggle.json or KAGGLE_USERNAME
    / KAGGLE_KEY env vars). Raises FileNotFoundError with setup hints if
    the CLI isn't installed or credentials are missing.

    Idempotent — if the archive already extracted under `dest` (presence
    of any subdir is the cheap heuristic), returns immediately.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if any(p.is_dir() for p in dest.iterdir()):
        logger.info("kaggle: cache present at %s; skipping download", dest)
        return dest

    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "kaggle CLI not found. Install with `pip install kaggle` "
            "and configure auth (~/.kaggle/kaggle.json with API token "
            "from https://www.kaggle.com/settings)."
        ) from e

    logger.info("kaggle: downloading %s → %s", dataset_id, dest)
    result = subprocess.run(
        ["kaggle", "datasets", "download", dataset_id, "-p", str(dest), "--unzip"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle download failed:\n  stdout: {result.stdout}\n  stderr: {result.stderr}"
        )

    # Some Kaggle datasets ship without --unzip honoring; fall back to a manual pass.
    for archive in dest.glob("*.zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
        archive.unlink()
    return dest


def md5_of_dir(root: Path) -> str:
    """Stable hash of a directory's *file listing* (paths + sizes). Used
    by prepare_dataset to detect "source already prepped, skip the
    whole remap+split pipeline" — content-hashing every byte of FER2013
    on each run would be wasteful."""
    h = hashlib.md5()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(str(p.stat().st_size).encode())
    return h.hexdigest()


def scan_imagefolder(folder: Path) -> pd.DataFrame:
    """Walk an imagefolder layout (root/<label>/*.{png,jpg,jpeg}) and
    return a DataFrame with columns ['path', 'src_label']. Paths are
    absolute so downstream phases can read them without re-resolving."""
    rows: list[dict[str, Any]] = []
    for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        for img in class_dir.iterdir():
            if img.suffix.lower() in _IMG_EXTS:
                rows.append({"path": str(img.resolve()), "src_label": class_dir.name})
    if not rows:
        raise FileNotFoundError(f"no images found under {folder} — wrong path?")
    return pd.DataFrame(rows)


def apply_remap(
    df: pd.DataFrame,
    remap: dict[str, str],
    class_names: list[str],
) -> pd.DataFrame:
    """Replace src_label with an integer `label` column per class_names.
    Rows whose src_label maps to DROP_SENTINEL (or is missing from remap)
    are removed entirely. Returns a new df with columns ['path', 'label']."""
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    keep_rows: list[dict[str, Any]] = []
    dropped: Counter = Counter()
    for row in df.to_dict("records"):
        src = row["src_label"]
        target = remap.get(src)
        if target is None or target == DROP_SENTINEL:
            dropped[src] += 1
            continue
        if target not in name_to_idx:
            raise ValueError(
                f"remap target {target!r} for src_label {src!r} "
                f"not in class_names {class_names}"
            )
        keep_rows.append({"path": row["path"], "label": name_to_idx[target]})
    if dropped:
        logger.info("remap: dropped %d rows by class: %s", sum(dropped.values()), dict(dropped))
    return pd.DataFrame(keep_rows)


def carve_val(train: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random split — fraction `val_fraction` of `train` becomes val.
    Stratification could land later; for FER2013's size this is fine."""
    shuffled = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(shuffled) * val_fraction)
    val   = shuffled.iloc[:n_val].reset_index(drop=True)
    train_out = shuffled.iloc[n_val:].reset_index(drop=True)
    return train_out, val


def compute_class_weights(labels: pd.Series, num_classes: int) -> list[float]:
    """Inverse-frequency weights normalised to mean 1.0 so loss scale
    is unchanged vs uniform weighting."""
    counts = labels.value_counts().reindex(range(num_classes), fill_value=0)
    inv = 1.0 / counts.replace(0, 1).astype(float)
    inv = inv / inv.mean()
    return [float(x) for x in inv.tolist()]


def write_split_csvs(splits: dict[str, pd.DataFrame], cache_dir: Path) -> dict[str, Path]:
    """Persist each split DataFrame as CSV under cache_dir. Returns the
    mapping the DatasetSpec.splits field expects."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, df in splits.items():
        dest = cache_dir / f"{name}.csv"
        df.to_csv(dest, index=False)
        out[name] = dest
    return out


# ---- high-level helpers used by every dataset module ----------------------


def try_load_cached(cache_dir: Path, source_dir: Path):
    """Return a cached DatasetSpec if one exists and the source hasn't
    changed; otherwise None. Lets a dataset module short-circuit the
    download+remap+split pipeline on warm re-runs.

    Imported lazily inside dataset modules — `from pipeline import ingest`
    then `ingest.try_load_cached(...)`.
    """
    from .framework.specs import DatasetSpec
    manifest = cache_dir / "manifest.json"
    if not (manifest.exists() and source_dir.exists()):
        return None
    prior = DatasetSpec.from_manifest(manifest)
    if prior.source_md5 == md5_of_dir(source_dir):
        logger.info("dataset cache hit at %s", cache_dir)
        return prior
    logger.info("dataset source md5 changed — re-prepping")
    return None


def finalize_dataset(
    *,
    name: str,
    cache_dir: Path,
    source_dir: Path,
    class_names: list[str],
    label_remap: dict[str, str] | None = None,
    train_dir: str = "train",
    test_dir:  str = "test",
    val_fraction: float = 0.10,
    val_seed: int = 42,
    class_weights: str = "auto",  # "auto" | "uniform"
):
    """Walk source/<train|test>/<class>/*.png, remap, split, write CSVs,
    return the DatasetSpec and persist its manifest. The single shared
    "everything after the fetch" pipeline — each dataset module just
    fetches its source and calls this.

    `label_remap=None` is identity (every class_name maps to itself).
    Pass an explicit remap when the source labels differ from
    class_names (e.g. FER2013 7-class → EmpathBot 6-class).
    """
    import json
    from .framework.specs import DatasetSpec

    train_raw = scan_imagefolder(source_dir / train_dir)
    test_raw  = scan_imagefolder(source_dir / test_dir)

    remap = label_remap or {n: n for n in class_names}
    train_remapped = apply_remap(train_raw, remap, class_names)
    test_remapped  = apply_remap(test_raw,  remap, class_names)

    train_df, val_df = carve_val(train_remapped, val_fraction=val_fraction, seed=val_seed)

    weights: list[float] | None
    if class_weights == "auto":
        weights = compute_class_weights(train_df["label"], num_classes=len(class_names))
    elif class_weights == "uniform":
        weights = None
    else:
        raise ValueError(
            f"class_weights={class_weights!r} not understood; expected 'auto' or 'uniform'"
        )

    splits = write_split_csvs(
        {"train": train_df, "val": val_df, "test": test_remapped},
        cache_dir,
    )

    spec = DatasetSpec(
        name=name,
        cache_dir=cache_dir,
        splits=splits,
        num_classes=len(class_names),
        class_names=class_names,
        class_weights=weights,
        source_md5=md5_of_dir(source_dir),
    )
    (cache_dir / "manifest.json").write_text(json.dumps(spec.to_manifest(), indent=2))
    logger.info(
        "dataset %s ready — train=%d val=%d test=%d",
        name, len(train_df), len(val_df), len(test_remapped),
    )
    return spec
