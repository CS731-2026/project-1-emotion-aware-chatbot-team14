# Training pipeline (v2)

One opinionated way to train a model on a dataset, log everything to
disk, and produce a checkpoint you can drop into the live app.

## TL;DR

```bash
# one-time
pip install -r requirements-training.txt    # torch, torchvision, timm, kaggle, pyyaml, pandas, pillow
mkdir -p ~/.kaggle && cp <your_kaggle.json> ~/.kaggle/   # for dataset downloads
chmod 600 ~/.kaggle/kaggle.json

# every run
python train.py experiments/fer2013__tiny_cnn__baseline.yaml
# → output/run/fer2013__tiny_cnn__baseline__<ts>/{checkpoints,metrics.jsonl,artifacts/}
```

## How a run is described

Three names, one experiment yaml:

```yaml
# experiments/fer2013__tiny_cnn__baseline.yaml
dataset:  fer2013        # → datasets/fer2013.yaml
model:    tiny_cnn       # → models/tiny_cnn.py
config:   baseline       # → configs/baseline.yaml
seed:     42
phases:   [setup, prepare_dataset, train]
```

Each name resolves to a file in its own directory. The pipeline runs
the phases in order against a fresh `output/run/<slug>__<timestamp>/`.

## Adding things

| What | Where | Contract |
|---|---|---|
| A new dataset | `datasets/<name>.yaml` | source + class_names + label_remap + splits |
| A new model | `models/<name>.py` | `build(num_classes) -> nn.Module` + `PREPROCESS` |
| A new train config | `configs/<name>.yaml` | epochs, batch_size, augment, loss, optimizer |
| A new loss / opt / aug | `training/{losses,optimizers,augmentations}.py` | one branch + one entry in SUPPORTED |
| A new experiment | `experiments/<name>.yaml` | reference the three above |
| A new phase | `pipeline/phases.py` + register in `pipeline/driver.PHASES` | `def phase(ctx: Context) -> None` |

## What a run produces

```
output/run/fer2013__tiny_cnn__baseline__20260522-235214/
  config.yaml             # snapshot of the resolved config
  metrics.jsonl           # one row per scalar log (train/val/test losses + accuracies)
  artifacts/
    dataset_used.json     # which DatasetSpec was loaded
    history.json          # per-epoch metrics
    final.json            # best/last/test summary
    setup.txt
  checkpoints/
    best.pth              # weights at best val_acc
    last.pth              # weights at end of training
```

Everything under `output/` is gitignored. To clear local state: `rm -rf output/`.

## What's NOT in here yet

- Dedicated `evaluate` phase — for now per-epoch val + final test eval
  are bundled inside `train`. The dedicated phase will load any
  checkpoint and produce confusion matrices, per-class metrics, error
  grids, etc. without retraining.
- Sweeps. Run one experiment at a time today; a `sweep.py` that walks
  a `(datasets × models × configs)` cross-product can land later.
- Leaderboard. `tools/leaderboard.py` will scan `output/run/*/artifacts/final.json`
  and print a comparison table.

## Models / datasets shipping right now

- `models/tiny_cnn.py` — 24k-param 3-conv CNN. Baseline for "does the pipeline run?".
- `datasets/fer2013.yaml` — Kaggle `msambare/fer2013`, 7-class → EmpathBot 6-class with disgust dropped.
