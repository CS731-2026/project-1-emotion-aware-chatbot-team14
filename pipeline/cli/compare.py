"""Scan output/run/ + output/eval/ and print a leaderboard.

Reads two sources for every row:
  - output/run/<slug>/artifacts/final.json + config.yaml — training metrics
  - output/run/<slug>/eval/<dataset>/summary.json       — auto-eval after train
  - output/eval/baseline__<id>/<dataset>/summary.json   — hand-trained baselines

Columns include the in-distribution eval acc (the empath / fer2013 test
split the model was trained on) AND the held-out OOD acc (fer2013_holdout
by default), plus the generalization gap between them.

Usage:
    python -m pipeline.cli.compare                   # everything
    python -m pipeline.cli.compare --filter empathbot
    python -m pipeline.cli.compare --sort-by ood_acc
    python -m pipeline.cli.compare --eval-dataset empath_test
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml


# Default columns: the dataset the model was trained on (in-distribution
# test split) + the held-out OOD dataset. Override --in-dataset / --ood-dataset
# to compare against different splits.
DEFAULT_OOD_DATASET = "fer2013_holdout"


@dataclass
class RunSummary:
    run_dir:    Path
    dataset:    str
    model:      str
    config:     str
    timestamp:  str
    test_acc:   float | None
    val_acc:    float | None
    best_epoch: int | None
    epochs:     int | None
    eval_acc:        dict[str, float | None] = field(default_factory=dict)  # dataset name → acc
    eval_macro_f1:   dict[str, float | None] = field(default_factory=dict)
    is_baseline:     bool = False

    @property
    def slug(self) -> str:
        return f"{self.dataset}__{self.model}__{self.config}"

    def in_dist_acc(self) -> float | None:
        """In-distribution eval acc: the auto-eval pass on the dataset the
        model was trained on. Falls back to artifacts/final.json test_acc
        if the eval phase didn't run (older runs)."""
        return self.eval_acc.get(self.dataset, self.test_acc)


def _parse_slug(name: str) -> tuple[str, str, str, str]:
    """Run dirs are <dataset>__<model>__<config>__<ts> with `__` separators."""
    parts = name.split("__")
    if len(parts) < 4:
        return (name, "", "", "")
    *rest, ts = parts
    while len(rest) < 3:
        rest.append("")
    return (rest[0], rest[1], "__".join(rest[2:]), ts)


def _load_eval_dir(eval_dir: Path) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Walk <run_dir>/eval/<dataset>/summary.json and return two dicts
    keyed by dataset name: {acc} and {macro_f1}."""
    accs: dict[str, float | None] = {}
    f1s: dict[str, float | None] = {}
    if not eval_dir.is_dir():
        return accs, f1s
    for ds_dir in sorted(eval_dir.iterdir()):
        summary = ds_dir / "summary.json"
        if not (ds_dir.is_dir() and summary.exists()):
            continue
        try:
            data = json.loads(summary.read_text())
        except Exception:
            continue
        accs[ds_dir.name] = data.get("acc")
        f1s[ds_dir.name] = data.get("macro_f1")
    return accs, f1s


def _read_run(run_dir: Path) -> RunSummary | None:
    """Parse one output/run/<slug>__<ts>/ dir into a RunSummary."""
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

    eval_acc, eval_f1 = _load_eval_dir(run_dir / "eval")

    return RunSummary(
        run_dir=run_dir, dataset=dataset, model=model, config=config,
        timestamp=ts,
        test_acc=float(test_acc) if test_acc is not None else None,
        val_acc=float(val_acc)   if val_acc  is not None else None,
        best_epoch=f.get("best_epoch"),
        epochs=epochs,
        eval_acc=eval_acc,
        eval_macro_f1=eval_f1,
    )


def _read_baseline(baseline_dir: Path) -> RunSummary | None:
    """Parse one output/eval/baseline__<id>/ dir into a RunSummary."""
    eval_acc, eval_f1 = _load_eval_dir(baseline_dir)
    if not eval_acc:
        return None
    baseline_id = baseline_dir.name.removeprefix("baseline__")

    index = baseline_dir / "index.json"
    ts = ""
    if index.exists():
        try:
            data = json.loads(index.read_text())
            ts = data.get("evaluated_at", "").replace(":", "").replace("-", "")[:15]
        except Exception:
            pass

    # Use baseline_id as dataset slot so its in_dist_acc() picks an entry
    # gracefully — but baselines weren't trained on a specific empath split
    # via this pipeline, so display them as "(baseline)" in the dataset column.
    return RunSummary(
        run_dir=baseline_dir, dataset="(baseline)", model=baseline_id,
        config="—", timestamp=ts,
        test_acc=None, val_acc=None, best_epoch=None, epochs=None,
        eval_acc=eval_acc, eval_macro_f1=eval_f1,
        is_baseline=True,
    )


def _format_ts(ts: str) -> str:
    """20260524-045745 → 05-24 04:57."""
    try:
        dt = datetime.strptime(ts, "%Y%m%d-%H%M%S")
        return dt.strftime("%m-%d %H:%M")
    except ValueError:
        return ts[:11] if ts else "—"


def _fmt_acc(v: float | None) -> str:
    return f"{v*100:6.2f}%" if v is not None else "   —   "


def _fmt_f1(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "  —  "


def _fmt_gap(in_acc: float | None, ood_acc: float | None) -> str:
    if in_acc is None or ood_acc is None:
        return "  —  "
    delta = ood_acc - in_acc
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta*100:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="output/run", help="run dirs root")
    ap.add_argument("--eval-root", default="output/eval",
                    help="baseline eval root (output of `make evaluate-baseline`)")
    ap.add_argument("--filter", default=None,
                    help="substring filter on '<dataset> <model> <config>'")
    ap.add_argument("--ood-dataset", default=DEFAULT_OOD_DATASET,
                    help=f"held-out OOD dataset to display (default: {DEFAULT_OOD_DATASET})")
    ap.add_argument("--sort-by", default="ood_acc",
                    choices=["ood_acc", "in_acc", "macro_f1", "gap", "date"],
                    help="sort key (default: ood_acc desc)")
    ap.add_argument("--top", type=int, default=None,
                    help="keep only the top N rows after sort")
    args = ap.parse_args()

    rows: list[RunSummary] = []

    run_root = Path(args.root)
    if run_root.is_dir():
        for d in sorted(run_root.iterdir()):
            if not d.is_dir():
                continue
            r = _read_run(d)
            if r is not None:
                rows.append(r)

    eval_root = Path(args.eval_root)
    if eval_root.is_dir():
        for d in sorted(eval_root.iterdir()):
            if not (d.is_dir() and d.name.startswith("baseline__")):
                continue
            r = _read_baseline(d)
            if r is not None:
                rows.append(r)

    if args.filter:
        needle = args.filter.lower()
        rows = [r for r in rows
                if needle in f"{r.dataset} {r.model} {r.config}".lower()]

    if not rows:
        print(f"no runs under {run_root} or baselines under {eval_root}",
              file=sys.stderr)
        return 1

    ood = args.ood_dataset
    sort_keys = {
        "ood_acc":  lambda r: (r.eval_acc.get(ood) or -1, r.timestamp),
        "in_acc":   lambda r: (r.in_dist_acc() or -1,    r.timestamp),
        "macro_f1": lambda r: (r.eval_macro_f1.get(ood) or -1, r.timestamp),
        "gap":      lambda r: (
            ((r.eval_acc.get(ood) or 0) - (r.in_dist_acc() or 0)), r.timestamp,
        ),
        "date":     lambda r: r.timestamp,
    }
    rows.sort(key=sort_keys[args.sort_by], reverse=args.sort_by != "date")

    if args.top:
        rows = rows[:args.top]

    w_ds = max(7, max(len(r.dataset) for r in rows))
    w_md = max(5, max(len(r.model)   for r in rows))
    w_cf = max(6, max(len(r.config)  for r in rows))

    header = (
        f"{'dataset':<{w_ds}}  {'model':<{w_md}}  {'config':<{w_cf}}  "
        f"{'in_acc':>7}  {'ood_acc':>7}  {'gap':>7}  "
        f"{'macro_f1':>8}  when"
    )
    print(header)
    print("-" * (len(header) + 4))
    for r in rows:
        in_acc = r.in_dist_acc()
        ood_acc = r.eval_acc.get(ood)
        macro_f1 = r.eval_macro_f1.get(ood)
        print(
            f"{r.dataset:<{w_ds}}  {r.model:<{w_md}}  {r.config:<{w_cf}}  "
            f"{_fmt_acc(in_acc)}  {_fmt_acc(ood_acc)}  {_fmt_gap(in_acc, ood_acc)}  "
            f"{_fmt_f1(macro_f1):>8}  {_format_ts(r.timestamp)}"
        )
    print(f"\n{len(rows)} row(s) shown — in_acc=in-distribution test acc; "
          f"ood_acc=held-out {ood}; gap=ood-in. sort={args.sort_by}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
