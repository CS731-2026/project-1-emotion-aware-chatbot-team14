"""Scan output/run/ and print a leaderboard.

Reads every run dir's artifacts/final.json + config.yaml, sorts by
test_acc (or `--sort-by val_acc|best_epoch|date`), prints a tight
fixed-width table. Useful when iterating on a model: did the last
five tweaks help?

Usage:
    python -m pipeline.cli.compare                   # everything
    python -m pipeline.cli.compare --filter empathbot
    python -m pipeline.cli.compare --filter fer2013 --top 10
    python -m pipeline.cli.compare --sort-by val_acc
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class RunSummary:
    run_dir: Path
    dataset: str
    model:   str
    config:  str
    timestamp: str
    test_acc: float | None
    val_acc:  float | None
    best_epoch: int | None
    epochs:   int | None

    @property
    def slug(self) -> str:
        return f"{self.dataset}__{self.model}__{self.config}"


def _parse_slug(name: str) -> tuple[str, str, str, str]:
    """Run dirs are <dataset>__<model>__<config>__<ts> with `__` separators."""
    parts = name.split("__")
    if len(parts) < 4:
        return (name, "", "", "")
    *rest, ts = parts
    while len(rest) < 3:
        rest.append("")
    return (rest[0], rest[1], "__".join(rest[2:]), ts)


def _read_one(run_dir: Path) -> RunSummary | None:
    final = run_dir / "artifacts" / "final.json"
    cfg   = run_dir / "config.yaml"
    if not final.exists():
        return None
    try:
        f = json.loads(final.read_text())
    except Exception:
        return None
    dataset, model, config, ts = _parse_slug(run_dir.name)

    test_acc = (f.get("test") or {}).get("acc")
    if test_acc is None:
        test_acc = f.get("test_acc")
    val_acc = (f.get("best_val") or {}).get("acc")
    if val_acc is None:
        val_acc = f.get("best_val_acc")

    epochs = None
    if cfg.exists():
        try:
            epochs = (yaml.safe_load(cfg.read_text()) or {}).get("train_cfg", {}).get("epochs")
        except Exception:
            pass

    return RunSummary(
        run_dir=run_dir, dataset=dataset, model=model, config=config,
        timestamp=ts,
        test_acc=float(test_acc) if test_acc is not None else None,
        val_acc=float(val_acc)   if val_acc  is not None else None,
        best_epoch=f.get("best_epoch"),
        epochs=epochs,
    )


def _format_ts(ts: str) -> str:
    """20260524-045745 → 05-24 04:57."""
    try:
        dt = datetime.strptime(ts, "%Y%m%d-%H%M%S")
        return dt.strftime("%m-%d %H:%M")
    except ValueError:
        return ts


def _fmt_acc(v: float | None) -> str:
    return f"{v*100:6.2f}%" if v is not None else "   —   "


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="output/run", help="run dirs root")
    ap.add_argument("--filter", default=None,
                    help="substring filter on '<dataset> <model> <config>'")
    ap.add_argument("--sort-by", default="test_acc",
                    choices=["test_acc", "val_acc", "date", "best_epoch"],
                    help="sort key (default: test_acc desc)")
    ap.add_argument("--top", type=int, default=None,
                    help="keep only the top N rows after sort")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"no run root at {root}", file=sys.stderr)
        return 1

    rows: list[RunSummary] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        r = _read_one(d)
        if r is None:
            continue
        if args.filter and args.filter.lower() not in \
           f"{r.dataset} {r.model} {r.config}".lower():
            continue
        rows.append(r)

    if not rows:
        print(f"no runs under {root}", file=sys.stderr)
        return 1

    sort_keys = {
        "test_acc":   lambda r: (r.test_acc or -1, r.timestamp),
        "val_acc":    lambda r: (r.val_acc  or -1, r.timestamp),
        "date":       lambda r: r.timestamp,
        "best_epoch": lambda r: (r.best_epoch or -1, r.timestamp),
    }
    rows.sort(key=sort_keys[args.sort_by], reverse=args.sort_by != "date")

    if args.top:
        rows = rows[:args.top]

    # Column widths derived from data
    w_ds = max(7, max(len(r.dataset) for r in rows))
    w_md = max(5, max(len(r.model)   for r in rows))
    w_cf = max(6, max(len(r.config)  for r in rows))

    print(f"{'dataset':<{w_ds}}  {'model':<{w_md}}  {'config':<{w_cf}}  {'test':>7}  {'val':>7}  {'@ep':>4}  when")
    print("-" * (w_ds + w_md + w_cf + 7 + 7 + 4 + 14 + 10))
    for r in rows:
        print(f"{r.dataset:<{w_ds}}  {r.model:<{w_md}}  {r.config:<{w_cf}}  "
              f"{_fmt_acc(r.test_acc)}  {_fmt_acc(r.val_acc)}  "
              f"{r.best_epoch if r.best_epoch is not None else '—':>4}  "
              f"{_format_ts(r.timestamp)}")
    print(f"\n{len(rows)} run(s) shown, sorted by {args.sort_by}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
