# Migrating from notebooks → pipeline

If you wrote one of the notebooks under `Notebooks/`, your training
procedure has been ported into `pipeline/`. This doc explains the
mental model, where your notebook went, and how to keep iterating
without ever opening Jupyter again.

## Overview — the whole picture in one screen

End-to-end lifecycle of a model on this project:

```
┌──────────────┐    ┌──────────────────┐    ┌────────────────┐    ┌──────────────┐    ┌────────────┐
│  notebook    │ →  │  pipeline port   │ →  │  training run  │ →  │   deploy     │ →  │  live app  │
│ (Notebooks/) │    │  (pipeline/)     │    │ (output/run/…) │    │  (models/)   │    │ (app/…)    │
└──────────────┘    └──────────────────┘    └────────────────┘    └──────────────┘    └────────────┘
   you wrote          we ported                make train          make deploy-model    make dev
   prose + cells      modules + funcs          → checkpoints +     → registry entry     EMOTION_MODEL_ID
                                                 plots + report      in models.yaml      in .env
```

Where each thing lives in the repo:

| What | Where | Owned by |
|---|---|---|
| **Notebooks** (research, ground truth for ports) | `Notebooks/` | you |
| **Pipeline ports** (your training procedure as code) | `pipeline/models/<m>/` `pipeline/datasets/<d>/` | you |
| **Hyperparameter presets** (fast/baseline/thorough) | `configs/<c>.py` | shared |
| **What to actually run** | `runs.yaml` (repo root) | shared |
| **Run output** (checkpoints, metrics, plots) | `output/run/<slug>__<ts>/` | generated, gitignored |
| **Deployed checkpoints** | `models/<id>/` | generated, gitignored |
| **Live app's model registry** | `application/model_service/models.yaml` | `make deploy-model` writes it |
| **Live app surface** | `application/{frontend,backend,model_service}/` | shared |
| **Bare emotion-bot test page** | `application/frontend/src/routes/emotion-test/` | shared |
| **Pipeline framework** (orchestration internals) | `pipeline/framework/` | you don't touch |
| **Shared training helpers** (losses, optimizers, augments, reporting) | `pipeline/training/` | extended occasionally |
| **Vendored third-party repos** (POSTER_V2) | `vendor/<repo>/` | git-weave manages |

The cycle you'll repeat every time you iterate on a model:

```
1.  edit pipeline/models/<m>/{model,augment,train_loop}.py
2.  add or modify a line in runs.yaml
3.  make train                             ← trains, writes output/run/<slug>__<ts>/
4.  inspect artifacts/{training_curves,confusion_matrix}.png + classification_report.txt
5.  make deploy-model RUN=LATEST ID=<id>   ← copies checkpoint + registers in models.yaml
6.  EMOTION_MODEL_ID=<id> make dev         ← live app loads it
7.  open http://localhost:5173/emotion-test/   ← isolated harness to verify behaviour
8.  make publish-model ID=<id>             ← share weights with team via Kaggle (no git binaries)
```

Three pieces are intentionally out of your hair:
- **`pipeline/framework/`** — Config, Context, DatasetSpec, the driver. The
  contract these expose to your model/dataset functions is the only thing
  you need; the internals are stable.
- **`pipeline/training/loop.py`** — `auto_device`, `train_one_epoch`,
  `evaluate`, `collect_predictions`, `merge_cfg`. Helpers your train_loop
  composes with.
- **`pipeline/training/reporting.py`** — `write_standard_artifacts(...)`.
  Call this once at the end of your training and you get the same plots +
  report your notebook produced.

**You're probably reading this because…**

- *"Where did my notebook go?"* → § 1 mapping table
- *"How do I run my port?"* → § 2 (`make train`)
- *"How do I tweak a hyperparameter without editing 5 files?"* → § 3 (runs.yaml `train_cfg:`) + § 4 (three layers)
- *"How do I get my trained model into the chat app?"* → § 7 (deploy-model)
- *"How do I test the model without the chat distracting me?"* → § 8 (/emotion-test/)
- *"How do I add a model from scratch?"* → § 9 (skeleton)
- *"How do I share weights without committing 200 MB binaries?"* → § 7 (publish-model / fetch-models)

## The mental model — three functions

Every training run in the pipeline is the composition of three things:

```
  dataset.prepare(ctx)  →  spec         (where do the images come from?)
  model.train(ctx, spec)  →  trained    (what gets trained, how?)
  config.CONFIG           →  hparams    (epochs, batch_size, lr, …)
```

Your notebook used to be *all three things in one file*. The pipeline
splits them so they can be re-combined and shared:

| You write | Lives at | What it exports |
|---|---|---|
| **A dataset function** | `pipeline/datasets/<name>/__init__.py` | `prepare(ctx) -> DatasetSpec` |
| **A model function** | `pipeline/models/<name>/__init__.py` | `train(ctx, dataset) -> TrainedModel` |
| **A config function** | `configs/<name>.py` | `CONFIG = {...}` dict |

A "run" is just a YAML line that picks one of each:

```yaml
# runs.yaml
runs:
  - { dataset: fer2013, model: empathbot_final, config: thorough }
```

Run with `make train` — done.

### How `runs.yaml` names become function calls

The names in `runs.yaml` are resolved via `importlib`. Each field
imports a Python module from a fixed location; the framework then
calls a fixed attribute on that module:

```
yaml field          module imported                      framework calls
─────────────────   ────────────────────────────────     ──────────────────────────────
dataset: my_data  → pipeline.datasets.my_data         → .prepare(ctx)  → DatasetSpec
model:   my_model → pipeline.models.my_model          → .train(ctx, dataset)
                                                          → TrainedModel
config:  thorough → configs.thorough                  → reads .CONFIG (dict)
                                                          and .NAME (slug)
```

So when you write a new model `my_model`:
- The folder name `pipeline/models/my_model/` is what you put in `runs.yaml`
- The function the framework will call is `pipeline.models.my_model.train`
- That's the function the tutorial's `__init__.py` exports as `train` —
  it's the contract between your code and the pipeline

Same on the dataset side: write `pipeline/datasets/my_data/__init__.py`
exporting `NAME`, `CLASS_NAMES`, and `prepare(ctx)`; reference it as
`dataset: my_data` in `runs.yaml`.

## Break your model code down

Your notebook is probably one long file. Your model **module** should
not be. The convention every existing model follows:

```
pipeline/models/<your_model>/
  __init__.py     # 1. thin surface: exports build() + train()
  model.py        # 2. nn.Module class — architecture only
  augment.py      # 3. TRAIN_TF / VAL_TF transforms
  data.py         # 4. Dataset class if it needs custom routing
  loss.py         # 5. custom loss if any (FocalLoss, etc)
  train_loop.py   # 6. CFG dict + run() that orchestrates training
```

Why split? Each file owns *one concern*. When a teammate later wants
to swap an augmentation or tweak the loss without touching the
training loop, they edit one small file instead of grepping a
500-line notebook.

### Read the tutorial modules first

The fastest way to learn the framework is to **read the two tutorial
modules top-to-bottom**. Both are working code that trains end-to-end
AND the source the scaffolder copies from. Every framework affordance
you'll use appears once, with small inline comments next to the code
they explain:

```
pipeline/models/tutorial/        ← the model-side reference
  __init__.py     ← start here. ~25 lines. The pipeline-facing surface.
  model.py        ← architecture. ResNet-18 + dropout head.
  augment.py      ← TRAIN_TF / VAL_TF, rationale per knob.
  data.py         ← CsvImageDataset — copy verbatim for most cases.
  train_loop.py   ← THE main file. Opens with a FRAMEWORK CHEATSHEET
                    listing every affordance (ctx surface, dataset
                    surface, training helpers, reporting, kaggle).

pipeline/datasets/tutorial/      ← the dataset-side reference
  __init__.py     ← one file. Opens with a DATASET CHEATSHEET listing
                    every ingest helper. Demonstrates cache check →
                    acquire (generate_synthetic / download_kaggle) →
                    label remap (with __drop__) → finalize_dataset.
```

Other models at increasing complexity once you've read the tutorial:

  - `pipeline/models/mlp/` — minimal; uses the shared classifier helper
  - `pipeline/models/resnet18/` — simpler custom loop, no per-sample routing
  - `pipeline/models/empathbot_final/` — full-featured: split-LR, freeze
    schedule, MixUp, label smoothing, neg_boost, per-class augmentation

### Starting from the scaffolder

```bash
make new-model ID=my_model                  # default: tutorial template
make new-model ID=my_model TEMPLATE=simple  # 30-line minimum (no custom loop)
```

The tutorial scaffolder copies `pipeline/models/tutorial/` with names
rewritten (`Tutorial` → `MyModel`, `tutorial` → `my_model`). You get
five files, all annotated, that trains on first run. Delete the
comments and tweak the architecture from there.

## 1. Find your port

Notebooks → pipeline modules:

| Your notebook | Lives now at |
|---|---|
| `1_dataset_pipeline.ipynb` | `pipeline/datasets/empath/` (sub-loaders: `affectnet.py`, `rafdb.py`, `sfew.py`, optional `face_crop.py`) |
| `2_benchmark_resnet18*.ipynb` (3 variants) | `pipeline/models/resnet18/` (same training procedure across all three) |
| `2_emotion-recognition-resnet18.ipynb` | `pipeline/models/resnet18_fer_onecycle/` |
| `3_benchmark_posterplus.ipynb` | `pipeline/models/posterplus/` (inference benchmark; expects published RAF-DB checkpoint) |
| `4_benchmark_ada_df.ipynb` | `pipeline/models/ada_df/` |
| `5_final_empathbot_training.ipynb` | `pipeline/models/empathbot_v3/` |
| `5_final_empathbot_training_v4.ipynb` | `pipeline/models/empathbot_final/` |
| `6_empathbot_v1_resnet18.ipynb` | `pipeline/models/empathbot_resnet18/` |
| `6b_empathbot_v1_improvements.ipynb` | `pipeline/models/empathbot_v1/` |
| `7_kash_dataset_prep.ipynb` | `pipeline/datasets/kash/` (optional `quality_filter.py`) |

Open the port's `__init__.py` — its docstring names the exact notebook
cells each `.py` file was lifted from. The notebook is still the
source of truth; if a tweak makes the port diverge, update the
notebook first then re-lift.

## 2. Run your port

```bash
make install-training       # one-time: pip deps + git-weave sync + POSTER_V2 stage
make train-list             # see what's declared in runs.yaml
make train                  # run every enabled entry
```

Run dirs land at `output/run/<dataset>__<model>__<config>__<ts>/`.

## 3. Change what runs — edit `runs.yaml`, not Python

`runs.yaml` at the repo root is the single answer to "what runs?". To
skip a row, comment it out OR add `enabled: false`:

```yaml
runs:
  - { dataset: synthetic_smoke, model: resnet18, config: fast }
  # - { dataset: fer2013, model: empathbot_final, config: thorough }
  - { dataset: fer2013, model: empathbot_v1, config: thorough, enabled: false }
```

To tweak a hyperparameter for **just one run** without forking a
config, add a `train_cfg:` block:

```yaml
runs:
  - dataset: fer2013
    model:   empathbot_final
    config:  thorough
    train_cfg:
      backbone_freeze_epochs: 3
      mixup_alpha: 0.3
```

The override layer is shallow-merged over the named config. Any key
already present in the model's `CFG` is overridable — no need to
register keys anywhere.

## 4. Where hyperparameters live (three layers)

| Layer | When to edit | File |
|---|---|---|
| **Model CFG** | Change the notebook defaults for everyone | `pipeline/models/<m>/train_loop.py` (top of file, `CFG = {...}`) |
| **Named config** | Cross-cutting preset (fast vs thorough) | `configs/<name>.py` (`CONFIG = {...}`) |
| **Per-run train_cfg** | Override for a single run | `runs.yaml` (`train_cfg:` block) |

Resolution: model CFG → named config → train_cfg. Whichever is most
specific wins.

## 5. What a run produces

Every run dir contains:

```
config.yaml                          resolved config snapshot
metrics.jsonl                        per-batch + per-epoch scalar log
checkpoints/best.pth                 best-val-acc checkpoint
artifacts/
  setup.txt                          repo + git state at run start
  dataset_used.json                  which CSVs the dataset module produced
  history.json                       per-epoch history
  final.json                         best_epoch, best_val, test_*
  training_curves.png                train/val loss + acc (+ lr)
  confusion_matrix.png               raw + row-normalised, side-by-side
  classification_report.txt          sklearn per-class P/R/F1
  per_class_metrics.json             same numbers, structured
```

These are the artifacts your notebook's final cells produced. If
something's missing, the post-training step logged a warning — search
the run's stdout for `reporting:`.

### Adding your own artifacts (any plot, table, metric you want)

Every phase function receives a `ctx` (Context). It exposes five
artifact-save methods — that's the entire API. **You never write a
path yourself** — `ctx` knows where the current run dir is and
handles the filesystem for you:

```python
def run(ctx, dataset, model):
    ...
    # any matplotlib figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    ctx.save_image("my_custom_plot", fig)          # → artifacts/my_custom_plot.png

    # any JSON-serialisable dict (or list of dicts, etc)
    ctx.save_json("eval_breakdown", {              # → artifacts/eval_breakdown.json
        "per_subject": {...},
        "by_lighting": {...},
    })

    # plain text — reports, prompts, debug dumps
    ctx.save_text("notes", "trained 30 epochs, …") # → artifacts/notes.txt

    # one metric line — appended to metrics.jsonl, cheap to call per batch
    ctx.save_scalar("custom/lr", current_lr, step=epoch)

    # torch state dict (you usually don't need this — write_standard_artifacts
    # already saves best.pth; this is for extra/intermediate checkpoints)
    ctx.save_checkpoint("epoch_10_snapshot", model.state_dict())
```

All paths land under the run dir automatically. **The
`output/run/<slug>__<ts>/` directory is an implementation detail —
you only touch it when reading completed runs back from disk
(`ctx.run_dir` is exposed if you really need it, e.g. to load the
best.pth back at end of training).**

Files with no extension get a default one (`.png` for save_image,
`.json` for save_json, `.txt` for save_text). Subdirectories work:
`ctx.save_image("epoch_5/predictions", fig)` lands at
`artifacts/epoch_5/predictions.png`. The parent dirs are created
automatically.

## 6. Datasets

Auto-downloadable:
- `synthetic_smoke`, `synthetic_imbalanced` — generated in `output/data/`, no network
- `fer2013` — Kaggle (`KAGGLE_USERNAME` + `KAGGLE_KEY` in `.env`, or `~/.kaggle/kaggle.json`)

Local-only (set env vars):
- `empath` — `EMPATH_AFFECTNET_DIR`, `EMPATH_RAFDB_DIR`, `EMPATH_SFEW_DIR`
  + optional `EMPATH_FACE_CROP=1` to run the notebook 1 YOLO crop step inline
- `kash` — `KASH_DATASET_DIR` (or place at `output/data/kash/raw/`)
  + optional `KASH_FACE_CROP=1`, `KASH_BLUR_FILTER=1`, `KASH_BLUR_THR=80.0`

All env vars can live in `.env` at the repo root — `pipeline/train.py`
calls `load_dotenv()` at startup.

## 7. Deploy a trained model into the live app

```bash
make deploy-model RUN=LATEST ID=my_empathbot
# or pick a specific run:
make deploy-model RUN=fer2013__empathbot_final__thorough__20260524-... ID=empathbot_final
```

That:
1. Copies `output/run/<RUN>/checkpoints/best.pth` → `models/<ID>/best.pth`
2. Adds an entry to `application/model_service/models.yaml`:
   ```yaml
   my_empathbot:
     path: models/my_empathbot/best.pth
     variant: empathbot
   ```

Then in `.env`:

```env
EMOTION_MODEL_ID=my_empathbot
```

`make dev` loads the new model on next service start.

Variants supported by the service: `placeholder`, `resnet18`,
`empathbot`. Pass `VARIANT=resnet18` to override the inferred one.

### Sharing weights with the team — "creds only"

Trained checkpoints don't belong in git. The team distributes them
through one Kaggle dataset (slug in `KAGGLE_WEIGHTS_SLUG`, default
`team14/empathbot-checkpoints`).

**Author flow** — you trained a model:

```bash
make deploy-model RUN=LATEST ID=my_empathbot     # local: copy + register
make publish-models                              # upload all local models/<id>/* to Kaggle
```

**Teammate flow** — they want to *use* your model in the live app:

```bash
# .env
KAGGLE_USERNAME=their_username
KAGGLE_KEY=their_api_key
EMOTION_MODEL_ID=my_empathbot

make dev
```

That's it. The model service auto-fetches the checkpoint from Kaggle
on first boot (when `models/my_empathbot/best.pth` is missing) and
caches it locally. Subsequent boots skip the fetch.

If your teammate prefers to pull everything explicitly:

```bash
make fetch-models           # pulls all team weights into models/
```

Notes:
- `publish-models` (plural) uploads **all** `models/<id>/*` as one
  Kaggle dataset version. Kaggle datasets are atomic, so per-model
  publish would erase every other team weight. The `publish-model`
  (singular) target still exists but warns; reach for it only if you
  know what you're doing.
- The slug + `KAGGLE_*` creds live in `.env`. `application/model_service/
  config.py` and `pipeline/train.py` both call `load_dotenv()` so a
  single .env covers training, deploy, publish, and the live service.
- For programmatic use (from a notebook or ad-hoc script):
  `from pipeline.kaggle import download_dataset, publish_models,
  fetch_models, creds_present`.

## 8. Test your deployed model in isolation

`make dev`, then open <http://localhost:5173/emotion-test/>.

A bare harness — webcam → face crop → emotion classifier. No chat, no
LLM, no transcripts. You can:
- Watch live emotion + confidence per frame
- Preview the face crop the model is actually seeing
- Flip `force_label` / `cycle_test_labels` / `log_predictions` from the
  sidebar to debug the path without restarting the service
- Track running prediction counts per class

The full app at `/` still works as normal.

## 9. Adding a new model from scratch

Mirror an existing folder, one file per concern (see § "Break your
model code down" above):

```
pipeline/models/<your_model>/
  __init__.py     # exports build(num_classes) + train(ctx, dataset)
  model.py        # nn.Module subclass — architecture only
  augment.py      # TRAIN_TF / VAL_TF transforms
  train_loop.py   # CFG (hyperparameters) + run(ctx, dataset, model)
```

Skeleton for the four files:

```python
# model.py
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self, num_classes: int): ...
    def forward(self, x): ...

def build(num_classes: int) -> nn.Module:
    return MyModel(num_classes=num_classes)
```

```python
# augment.py
import torchvision.transforms as T
TRAIN_TF = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(...)])
VAL_TF   = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(...)])
```

```python
# train_loop.py
from pipeline.training.loop import auto_device, collect_predictions, merge_cfg
from pipeline.training.reporting import write_standard_artifacts

CFG = dict(epochs=40, batch_size=32, lr=1e-4, weight_decay=1e-4)

def run(ctx, dataset, model):
    cfg = merge_cfg(CFG, ctx.config.train_cfg)
    device = auto_device()
    # ... your training loop ...
    # at the end:
    test_preds, test_labels = collect_predictions(model, test_loader, device)
    write_standard_artifacts(ctx, history=history,
        test_preds=test_preds, test_labels=test_labels,
        num_classes=dataset.num_classes, class_names=dataset.class_names,
        final_summary={...})
```

```python
# __init__.py
from .model import build
from .train_loop import run as _run

def train(ctx, dataset):
    return _run(ctx, dataset, model=build(dataset.num_classes))
```

Then add a line to `runs.yaml`:

```yaml
runs:
  - { dataset: fer2013, model: my_model, config: thorough }
```

If your training procedure is generic (CE + AdamW), use
`pipeline.training.standard.train_classifier` instead of writing
`train_loop.py` — see `pipeline/models/mlp/__init__.py` for the
minimal pattern (~30 lines total).

## 10. Adding a new dataset

```
pipeline/datasets/<your_dataset>/
  __init__.py     # NAME, CLASS_NAMES, prepare(ctx) -> DatasetSpec
```

`prepare()` returns a `DatasetSpec` (see `pipeline/framework/specs.py`)
with three CSV paths (train/val/test), each row `path,label`. The
shared `ingest` helpers (`download_kaggle`, `scan_imagefolder`,
`apply_remap`, `carve_val`, `finalize_dataset`) handle the common
patterns — see `pipeline/datasets/fer2013/__init__.py` for the simple
case (Kaggle download + standard split).

## 11. Adding a new config

`configs/<name>.py` exporting `NAME` + `CONFIG`:

```python
NAME = "my_config"
CONFIG = {
    "epochs": 20,
    "batch_size": 64,
    # any key that exists in a model's CFG can go here and will
    # override that model's default
}
```

## Questions

- Audit of port fidelity: PR #21 description + the in-conversation audit
- Pipeline architecture: `pipeline/framework/{store,config,context,specs}.py`
  + their docstrings
- What live debug flags exist: `application/model_service/core/debug_flags.py`
