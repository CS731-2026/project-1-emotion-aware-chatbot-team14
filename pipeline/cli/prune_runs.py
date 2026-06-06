"""Prune old run dirs under output/run/, keeping the newest per (dataset, model, config).

Run history accumulates across iteration — every `make train` produces a
new timestamped run dir, and rerunning the same (dataset, model, config)
combo many times leaves the disk full of redundant checkpoints. This CLI
groups runs by their slug (everything before the trailing __<timestamp>),
keeps the newest run per group, and offers to delete the older ones.

**Dry-run by default** — invoking without --apply just lists what would
be deleted (size + count). Pass --apply to actually delete. Never runs
automatically; you must invoke it manually.

Usage:
    make prune-runs                       # preview only
    make prune-runs APPLY=1                # actually delete

    python -m pipeline.cli.prune_runs                  # preview
    python -m pipeline.cli.prune_runs --apply          # delete
    python -m pipeline.cli.prune_runs --keep 2 --apply # keep newest 2 per combo
    python -m pipeline.cli.prune_runs --filter empath  # only consider matching runs

What's safe:
  * Each run dir is independent — checkpoints, artifacts, eval bundles all
    live inside it. Deleting a run dir loses *only* that run's results,
    nothing shared.
  * The newest run per combo always survives (default --keep 1).
  * Baselines under output/eval/baseline__* are untouched.
  * Source data under output/data/ is untouched.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def _parse_slug(name: str) -> tuple[str, str]:
    """Run dirs are <dataset>__<model>__<config>__<timestamp>.

    Returns (combo_slug, timestamp). combo_slug = "<dataset>__<model>__<config>"
    is the grouping key; timestamp orders within a group.
    """
    parts = name.rsplit("__", 1)
    if len(parts) != 2:
        return (name, "")
    return (parts[0], parts[1])


def _dir_size(path: Path) -> int:
    """Sum of file sizes under `path`. Symlinks not followed."""
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:6.1f} TB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="output/run",
                    help="run dirs root (default: output/run)")
    ap.add_argument("--keep", type=int, default=1,
                    help="how many newest runs per (dataset, model, config) "
                         "combo to keep (default: 1)")
    ap.add_argument("--filter", default=None,
                    help="only consider run dirs whose slug contains this "
                         "substring (case-insensitive)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete. Without this flag, just lists "
                         "what would be deleted (DRY-RUN, the default).")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"✗ {root} does not exist — nothing to prune.", file=sys.stderr)
        return 1
    if args.keep < 1:
        print(f"✗ --keep must be ≥ 1 (got {args.keep})", file=sys.stderr)
        return 2

    # Group all run dirs by combo slug.
    groups: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    needle = args.filter.lower() if args.filter else None
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if needle and needle not in d.name.lower():
            continue
        combo, ts = _parse_slug(d.name)
        if not ts:
            # Doesn't match the expected <slug>__<ts> shape — leave it alone
            # rather than risk deleting something we don't understand.
            continue
        groups[combo].append((ts, d))

    if not groups:
        print(f"no run dirs found under {root}"
              + (f" matching --filter={args.filter!r}" if needle else ""),
              file=sys.stderr)
        return 0

    # Within each group, sort newest-first by timestamp string (lexicographic
    # works since timestamps are zero-padded YYYYMMDD-HHMMSS).
    to_keep: list[Path] = []
    to_delete: list[Path] = []
    for combo, runs in groups.items():
        runs.sort(key=lambda t: t[0], reverse=True)
        keep = runs[:args.keep]
        drop = runs[args.keep:]
        to_keep.extend(p for _, p in keep)
        to_delete.extend(p for _, p in drop)

    if not to_delete:
        print(f"nothing to prune — every combo already has ≤ {args.keep} run(s)")
        return 0

    # Group output for readability.
    by_combo: dict[str, list[Path]] = defaultdict(list)
    for d in to_delete:
        combo, _ = _parse_slug(d.name)
        by_combo[combo].append(d)

    total_bytes = 0
    print(f"{'DELETING' if args.apply else 'DRY-RUN — would delete'} "
          f"{len(to_delete)} run dir(s) across {len(by_combo)} combo(s) "
          f"(keep newest {args.keep} per combo):")
    print()
    for combo in sorted(by_combo):
        print(f"  [{combo}]")
        for d in sorted(by_combo[combo]):
            size = _dir_size(d)
            total_bytes += size
            print(f"    {_fmt_bytes(size)}   {d.name}")
        print()
    print(f"total: {_fmt_bytes(total_bytes)} across {len(to_delete)} dir(s)")

    if not args.apply:
        print()
        print("DRY-RUN — no files were touched. Re-run with --apply to delete:")
        print(f"    make prune-runs APPLY=1"
              + (f"  FILTER={args.filter}" if args.filter else "")
              + (f"  KEEP={args.keep}"     if args.keep != 1 else ""))
        return 0

    deleted = errors = 0
    for d in to_delete:
        try:
            shutil.rmtree(d)
            deleted += 1
        except OSError as e:
            print(f"  ✗ failed to delete {d}: {e}", file=sys.stderr)
            errors += 1

    print()
    print(f"✓ deleted {deleted} run dir(s), freed ~{_fmt_bytes(total_bytes)}"
          + (f", {errors} error(s)" if errors else ""))
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
