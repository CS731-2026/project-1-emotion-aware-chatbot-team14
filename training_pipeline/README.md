# ML Pipeline Harness

A minimal ML experimentation harness. Handles CLI, config merging, per-step persistence, and resumption. Your pipeline imports it as a library — the harness has no opinion on your folder structure.

---

## Concepts

```
your run.py                     harness
───────────────                 ───────
import harness       →          library, knows nothing about your project

define routines

harness.main(routines)  →       parses CLI, merges configs, calls your routine

routine(config):
    harness.run(steps)  →       executes steps, manages store, persists artifacts
         └── step(store, config)
```

**Store** — the only way steps pass data to each other. In-memory during a run; serialised to disk after each step so runs can be resumed.

**Step** — a plain function `(store, config) -> Success | Failure`. One unit of work.

**Routine** — defines what steps run and in what order. Owns `runs_dir`.

---

## Quickstart

```python
# run.py
import harness
from my_steps import load_data, train, evaluate

def my_routine(config: dict) -> None:
    harness.run(
        steps=[load_data, train, evaluate],
        name="my_routine",
        config=config,
        runs_dir=Path("runs"),
    )

if __name__ == "__main__":
    harness.main(routines={"my_routine": my_routine})
```

```bash
python run.py --routine my_routine --config base.yaml --config debug.yaml
```

Multiple `--config` files are deep-merged left to right — later files override earlier keys.

---

## The Store

The store is how steps share data. It raises loudly on missing keys — no silent `None` returns.

```python
from harness import Success
from harness.store import Store

def load_data(store: Store, config: dict):
    dataset = build_dataset(config["n_samples"])
    store.put("dataset", dataset)
    return Success

def train(store: Store, config: dict):
    dataset = store.get("dataset")   # KeyError if missing — fail fast
    model = build_model(config)
    store.put("model", model)
    return Success
```

`store.put` overwrites silently — steps own their keys by convention.

The store is cleared and reloaded from disk before each step, so every step always sees a consistent snapshot of all prior completed steps.

---

## Steps

A step is a plain Python function:

```python
def my_step(store: Store, config: dict):
    value = store.get("input_key")
    result = do_work(value, config["param"])
    store.put("output_key", result)
    return Success          # or Failure("reason")
```

- Uncaught exceptions are caught by `run()` and become `Failure` automatically
- Steps do not need to handle their own exceptions unless they want to recover
- `run()` halts on the first `Failure` and does not continue

---

## Returning Success or Failure

```python
from harness import Success, Failure

def my_step(store, config):
    if something_wrong:
        return Failure("description of what went wrong")
    store.put("result", value)
    return Success
```

---

## Output Folders

Define named output directories in a setup step using `make_folders`. These are the public-facing outputs of your run — checkpoints, artifacts, evaluation results.

```python
from harness import make_folders, Success

def setup_outputs(store, config):
    folders = make_folders(store.get("run_dir"), {
        "checkpoints": "checkpoints",
        "artifacts":   "artifacts",
    })
    store.put("folders", folders)
    return Success
```

Write and read artifacts in later steps:

```python
def save_checkpoint(store, config):
    folders = store.get("folders")
    model = store.get("model")
    folders.save("checkpoints", "best", model.state_dict())
    return Success

def evaluate(store, config):
    folders = store.get("folders")
    state_dict = folders.load("checkpoints", "best")
    model.load_state_dict(state_dict)
    ...
```

`folders.save` and `folders.load` use the same type-aware serialiser registry as the store.

---

## Log Channels

Define named log files in a setup step. Steps write to whichever channel is semantically correct.

```python
from harness import make_log_channels, Success

def setup_outputs(store, config):
    logs = make_log_channels(store.get("run_dir"), {
        "training": "logs/training.log",
        "eval":     "logs/eval.log",
    })
    store.put("logs", logs)
    return Success
```

```python
def train_epoch(store, config):
    logs = store.get("logs")
    logs.training(f"epoch {epoch}  loss={loss:.4f}  acc={acc:.3f}")
    return Success
```

There is always a built-in `pipeline` channel written by `run()` itself — step timing, outcomes, failures.

---

## Compound Steps

A compound step is a factory that returns `(store, config) -> Success | Failure`. `run()` treats it as a single step.

```python
from harness import Failure, Success

def epoch_loop(*steps):
    def _run(store, config):
        for epoch in range(config["epochs"]):
            store.put("epoch", epoch)
            for step in steps:
                outcome = step(store, config)
                if isinstance(outcome, Failure):
                    return outcome
        return Success
    _run.__name__ = "epoch_loop"
    return _run
```

```python
harness.run(steps=[
    setup_device,
    setup_outputs,
    load_data,
    epoch_loop(train_epoch, validate, save_best_checkpoint),
    evaluate,
])
```

---

## Serialisation

Store values are serialised automatically after each step. Type dispatch:

| Type | Format |
|------|--------|
| `torch.Tensor` / `nn.Module` | `.pt` |
| `numpy.ndarray` | `.npy` |
| JSON-serialisable (dict, list, int, float, str) | `.json` |
| Anything else | `.pkl` |

Register a custom serialiser for types that need special handling:

```python
from harness import register_serializer

register_serializer(
    MyType,
    save=lambda path, obj: ...,
    load=lambda path: ...,
    suffix=".mytype",
)
```

**DataLoaders should never go in the store** — construct them inside the step that needs them. Store the dataset instead.

---

## Resumption

If a run is interrupted, restart it with the same command. `run()` checks each step folder for a `completed` marker and skips steps that already finished, reloading their artifacts from disk. No extra arguments needed.

---

## Run Directory Layout

```
runs/my_routine_2026-03-04_14-30/
  config.json               ← final merged config
  logs/
    pipeline.log            ← infrastructure log (always present)
    training.log            ← routine-defined channels
  steps/
    001_setup_device/
      device.json
      config.json
      pipeline.log
      completed
    002_load_data/
      dataset.pkl
      config.json
      pipeline.log
      completed
    ...
  checkpoints/              ← routine-defined output folders
  artifacts/
```

---

## Custom Serialisers — `FolderSet` and `LogChannels`

`FolderSet` and `LogChannels` are harness infrastructure types. They are automatically registered with their own serialisers (`.folderset` / `.logchannels`) when you `import harness`, so they survive the store clear/reload cycle between steps with no extra work.

---

## Example Pipeline

See [example/](example/) for a complete working pipeline:

```bash
python example/run.py --routine train_classifier \
    --config example/base.yaml \
    --config example/debug.yaml
```

- [example/pipeline.py](example/pipeline.py) — model, steps, epoch loop
- [example/run.py](example/run.py) — entry point
- [example/base.yaml](example/base.yaml) / [example/debug.yaml](example/debug.yaml) — configs
