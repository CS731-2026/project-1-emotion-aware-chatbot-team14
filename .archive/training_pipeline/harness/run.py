from __future__ import annotations

import json
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .log_channels import LogChannel, LogChannels
from .serialization import load_value, save_value
from .store import Store
from .types import Failure, Loop, Success, _Success

Step = Callable[[Store, dict], "_Success | Failure"]


def run(
    steps: list,
    name: str,
    config: dict,
    runs_dir: Path,
    overrides: dict | None = None,
) -> list[dict]:
    """Execute a sequence of steps with full per-step persistence and resumption.

    steps may contain plain callables or Loop objects (created via harness.loop()).
    Loop objects produce hierarchical step folders and are serialised by the
    harness after every inner step — not just at the end of the loop.

    Returns:
        Flat list of per-step result dicts (includes all inner steps):
        [{"step": name, "status": "ok" | "skipped" | "failure", ...}, ...]
    """
    if overrides:
        config = {**config, **overrides}

    # Seed RNG
    seed = config.get("seed", 0)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = runs_dir / f"{name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    pipeline_log = LogChannel(run_dir / "logs" / "pipeline.log")
    pipeline_log(f"run started: {name}")
    pipeline_log(f"run directory: {run_dir}")

    steps_dir = run_dir / "steps"
    steps_dir.mkdir(exist_ok=True)

    store = Store()
    store.put("run_dir", run_dir)
    results: list[dict] = []

    _execute_steps(
        steps=steps,
        store=store,
        config=config,
        parent_dir=steps_dir,
        root_steps_dir=steps_dir,
        run_dir=run_dir,
        pipeline_log=pipeline_log,
        results=results,
        depth=0,
    )

    pipeline_log("run finished")
    pipeline_log("results: " + ", ".join(f"{r['step']}={r['status']}" for r in results))
    return results


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------

def _execute_steps(
    steps: list,
    store: Store,
    config: dict,
    parent_dir: Path,
    root_steps_dir: Path,
    run_dir: Path,
    pipeline_log: LogChannel,
    results: list[dict],
    depth: int,
) -> "_Success | Failure":
    """Execute a list of steps sequentially. Steps may be callables or Loop objects."""
    for idx, step in enumerate(steps):
        if isinstance(step, Loop):
            outcome = _execute_loop(
                loop=step, idx=idx, store=store, config=config,
                parent_dir=parent_dir, root_steps_dir=root_steps_dir,
                run_dir=run_dir, pipeline_log=pipeline_log,
                results=results, depth=depth,
            )
        else:
            outcome = _execute_step(
                step=step, idx=idx, store=store, config=config,
                parent_dir=parent_dir, root_steps_dir=root_steps_dir,
                run_dir=run_dir, pipeline_log=pipeline_log,
                results=results, depth=depth,
            )

        if isinstance(outcome, Failure):
            return outcome

    return Success


def _execute_step(
    step: Callable,
    idx: int,
    store: Store,
    config: dict,
    parent_dir: Path,
    root_steps_dir: Path,
    run_dir: Path,
    pipeline_log: LogChannel,
    results: list[dict],
    depth: int,
) -> "_Success | Failure":
    pad = "  " * depth
    step_name = getattr(step, "__name__", f"step_{idx:03d}")
    step_dir = parent_dir / f"{idx + 1:03d}_{step_name}"

    # --- Resumption check ---
    if (step_dir / "completed").exists():
        pipeline_log(f"[{step_name}] skipped (already completed)")
        print(f"{pad}  skip  {step_name}")
        _load_step_artifacts(step_dir, store)
        results.append({"step": step_name, "status": "skipped"})
        return Success

    # --- Pre-step: clear and reload all prior completed artifacts ---
    store.clear()
    store.put("run_dir", run_dir)
    _reload_up_to(root_steps_dir, step_dir, store)

    step_dir.mkdir(parents=True, exist_ok=True)

    # --- Execute ---
    pipeline_log(f"[{step_name}] starting")
    print(f"{pad}  run   {step_name}")
    start = datetime.now()

    try:
        outcome = step(store, config)
        if outcome is None:
            outcome = Success
    except Exception as exc:
        outcome = Failure(reason=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    elapsed = (datetime.now() - start).total_seconds()

    if isinstance(outcome, _Success):
        _save_step_artifacts(step_dir, store)
        _save_step_config(step_dir, config)
        _snapshot_logs(step_dir, run_dir)
        (step_dir / "completed").touch()

        pipeline_log(f"[{step_name}] ok ({elapsed:.2f}s)")
        print(f"{pad}         ok  ({elapsed:.2f}s)")
        results.append({"step": step_name, "status": "ok"})
    else:
        reason = outcome.reason if isinstance(outcome, Failure) else "unknown"
        pipeline_log(f"[{step_name}] FAILED ({elapsed:.2f}s): {reason}")
        print(f"{pad}         FAILED  ({elapsed:.2f}s)")
        print(f"{pad}         reason: {reason}")
        results.append({"step": step_name, "status": "failure", "reason": reason})

    return outcome


def _execute_loop(
    loop: Loop,
    idx: int,
    store: Store,
    config: dict,
    parent_dir: Path,
    root_steps_dir: Path,
    run_dir: Path,
    pipeline_log: LogChannel,
    results: list[dict],
    depth: int,
) -> "_Success | Failure":
    pad = "  " * depth
    loop_name = loop.__name__
    loop_dir = parent_dir / f"{idx + 1:03d}_{loop_name}"

    # --- Resumption: entire loop already done ---
    if (loop_dir / "completed").exists():
        pipeline_log(f"[{loop_name}] skipped (already completed)")
        print(f"{pad}  skip  {loop_name}")
        _load_step_artifacts(loop_dir, store)
        results.append({"step": loop_name, "status": "skipped"})
        return Success

    loop_dir.mkdir(parents=True, exist_ok=True)
    # Marker so _reload_up_to can identify this as a loop container
    (loop_dir / "loop").touch()

    n = loop.n(config) if callable(loop.n) else loop.n
    print(f"{pad}  run   {loop_name}  (n={n})")
    pipeline_log(f"[{loop_name}] starting (n={n})")
    start_loop = datetime.now()

    outcome: "_Success | Failure" = Success
    i = 0
    while i < n:
        iter_dir = loop_dir / f"{loop.iter_name}_{i:03d}"

        # --- Skip completed iterations ---
        if (iter_dir / "completed").exists():
            print(f"{pad}    skip  {loop.iter_name} {i}")
            i += 1
            continue

        iter_dir.mkdir(parents=True, exist_ok=True)
        # Write iteration counter so _reload_up_to can inject it during resume
        (iter_dir / "iter.json").write_text(
            json.dumps({loop.iter_name: i}), encoding="utf-8"
        )

        print(f"{pad}    {loop.iter_name} {i}")

        outcome = _execute_steps(
            steps=loop.steps,
            store=store,
            config=config,
            parent_dir=iter_dir,
            root_steps_dir=root_steps_dir,
            run_dir=run_dir,
            pipeline_log=pipeline_log,
            results=results,
            depth=depth + 2,
        )

        if isinstance(outcome, Failure):
            break

        (iter_dir / "completed").touch()
        i += 1

        # Check pipeline-defined continue condition after each completed iteration
        if loop.while_ is not None and not loop.while_(store, config):
            break

    elapsed_loop = (datetime.now() - start_loop).total_seconds()

    if isinstance(outcome, _Success):
        # Save final store state to loop_dir so skip-on-resume works via
        # _load_step_artifacts(loop_dir, store) without recursing into iters
        _save_step_artifacts(loop_dir, store)
        _save_step_config(loop_dir, config)
        _snapshot_logs(loop_dir, run_dir)
        (loop_dir / "completed").touch()

        pipeline_log(f"[{loop_name}] ok ({elapsed_loop:.2f}s)")
        print(f"{pad}         ok  ({elapsed_loop:.2f}s)")
        results.append({"step": loop_name, "status": "ok"})
    else:
        reason = outcome.reason if isinstance(outcome, Failure) else "unknown"
        pipeline_log(f"[{loop_name}] FAILED ({elapsed_loop:.2f}s): {reason}")
        print(f"{pad}         FAILED  ({elapsed_loop:.2f}s)")
        results.append({"step": loop_name, "status": "failure", "reason": reason})

    return outcome


# ---------------------------------------------------------------------------
# Hierarchical reload
# ---------------------------------------------------------------------------

def _reload_up_to(search_dir: Path, stop_at: Path, store: Store) -> bool:
    """Walk search_dir in sorted order, loading completed leaf-step artifacts.

    Traverses the full step hierarchy recursively. Stops — without loading —
    the moment it reaches stop_at. Returns True if stop_at was found.

    Three kinds of directory are recognised by marker files:
      - Loop container  has a "loop" file
      - Iteration dir   has an "iter.json" file  (written by _execute_loop)
      - Leaf step       has a "completed" file (and no loop/iter.json markers)
    """
    for entry in sorted(search_dir.iterdir()):
        if not entry.is_dir():
            continue

        if entry == stop_at:
            return True

        if (entry / "loop").exists():
            # Loop container
            if (entry / "completed").exists():
                # Fully done — load its saved final state directly (fast path)
                _load_step_artifacts(entry, store)
            else:
                # Partially done — recurse to find stop_at inside
                if _reload_up_to(entry, stop_at, store):
                    return True

        elif (entry / "iter.json").exists():
            # Iteration dir — inject the iteration counter, then recurse
            iter_data = json.loads((entry / "iter.json").read_text(encoding="utf-8"))
            for k, v in iter_data.items():
                store.put(k, v)
            if _reload_up_to(entry, stop_at, store):
                return True

        elif (entry / "completed").exists():
            # Leaf step — load its artifacts
            _load_step_artifacts(entry, store)

    return False


# ---------------------------------------------------------------------------
# Artifact helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _save_step_artifacts(step_dir: Path, store: Store) -> None:
    for key in store.keys():
        if key == "run_dir":
            continue
        value = store.get(key)
        try:
            save_value(step_dir / key, value)
        except Exception as exc:
            raise type(exc)(f"Failed to serialise store key {key!r}: {exc}") from exc


def _load_step_artifacts(step_dir: Path, store: Store) -> None:
    skip = {"completed", "config.json", "loop", "iter.json"}
    suffixes = {".pt", ".npy", ".json", ".pkl", ".folderset", ".logchannels"}

    seen_stems: set[str] = set()
    for file in sorted(step_dir.iterdir()):
        if file.name in skip:
            continue
        if file.name.endswith(".log"):
            continue
        if file.suffix in suffixes and file.stem not in seen_stems:
            seen_stems.add(file.stem)
            try:
                value = load_value(step_dir / file.stem)
                store.put(file.stem, value)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load artifact {file} during resume: {exc}"
                ) from exc


def _save_step_config(step_dir: Path, config: dict) -> None:
    with open(step_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def _snapshot_logs(step_dir: Path, run_dir: Path) -> None:
    import shutil
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        return
    for log_file in logs_dir.iterdir():
        if log_file.suffix == ".log":
            shutil.copy2(log_file, step_dir / log_file.name)
