# Migrating from notebooks → pipeline

If you wrote one of the notebooks under `Notebooks/`, your training
procedure has been ported into `pipeline/`. This is the short version
of where it went, how to verify it still does what your notebook did,
how to tweak it, and how to ship a trained model into the live app.

## 1. Find your port

Notebooks → pipeline modules:

| Your notebook | Lives now at |
|---|---|
| `1_dataset_pipeline.ipynb` | `pipeline/datasets/empath/` (sub-loaders: `affectnet.py`, `rafdb.py`, `sfew.py`, optional `face_crop.py`) |
| `2_benchmark_resnet18.ipynb`, `2_benchmark_resnet18_colab.ipynb`, `2_colab_hf_datasets_resnet18.ipynb` | `pipeline/models/resnet18/` (same training procedure across all three) |
| `2_emotion-recognition-resnet18.ipynb` | `pipeline/models/resnet18_fer_onecycle/` |
| `3_benchmark_posterplus.ipynb` | `pipeline/models/posterplus/` (inference benchmark; expects published RAF-DB checkpoint at `output/models/posterv2_rafdb.pth`) |
| `4_benchmark_ada_df.ipynb` | `pipeline/models/ada_df/` |
| `5_final_empathbot_training.ipynb` | `pipeline/models/empathbot_v3/` |
| `5_final_empathbot_training_v4.ipynb` | `pipeline/models/empathbot_final/` |
| `6_empathbot_v1_resnet18.ipynb` | `pipeline/models/empathbot_resnet18/` |
| `6b_empathbot_v1_improvements.ipynb` | `pipeline/models/empathbot_v1/` |
| `7_kash_dataset_prep.ipynb` | `pipeline/datasets/kash/` (optional `quality_filter.py`) |

Open the port's `__init__.py` — its docstring names the exact notebook
cells each `.py` file was lifted from. The notebook is still the
source of truth; if a tweak makes the port diverge from the notebook,
update the notebook first then re-lift.

## 2. Verify the port matches your notebook

Each port preserves the *training procedure* (architecture,
augmentations, loss, optimizer, scheduler, freeze schedule, MixUp gate,
checkpoint envelope). If a hyperparameter looks wrong:

```bash
# show the CFG block your model uses
grep -A 20 "^CFG =" pipeline/models/<your_model>/train_loop.py
```

Compare against the corresponding cell in your notebook (typically the
"hyperparameters" cell + the optimizer/scheduler cell). The port's
docstring at the top names the specific cells the values came from.

The audit at PR #21 walks each notebook → port pair and flags
discrepancies — read that first if something looks off.

## 3. Run your port

```bash
make install-training       # one-time: pip deps + git-weave sync + POSTER_V2 stage
make train-list             # see what's declared
make train                  # run every (dataset, model, config) triple
```

To skip every run except yours, comment out everything else in
`pipeline/train.py::RUNS`, or invoke a tighter subset inline:

```python
python -c "
import logging; logging.basicConfig(level=logging.INFO, format='%(message)s')
import configs.thorough as cfg
from pipeline.datasets import fer2013
from pipeline.models import empathbot_v1
from pipeline.driver import sweep
sweep([(fer2013, empathbot_v1, cfg)], fail_fast=True)
"
```

Run dir lands at `output/run/<dataset>__<model>__<config>__<ts>/`.

## 4. What your run produces

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
  training_curves.png                train/val loss + acc (+ lr if logged)
  confusion_matrix.png               raw + row-normalised, side-by-side
  classification_report.txt          sklearn per-class P/R/F1
  per_class_metrics.json             same numbers, structured
```

These are the artifacts your notebook's plots + reports produced. If
something's missing, the post-training step logged a warning — search
the run's stdout for `reporting:`.

## 5. Tweaking hyperparameters

Three layers:

| Where | When |
|---|---|
| `pipeline/models/<m>/train_loop.py::CFG` | Default values your notebook chose. Edit if the new defaults are better. |
| `configs/{fast,baseline,thorough}.py` | Cross-cutting overrides (epochs, batch_size). Affect every model that uses that config. |
| `ctx.config.train_cfg` (per run) | Per-run overrides — pass via a custom config or ad-hoc kwargs. |

`_config_overrides()` at the top of each train_loop merges
`ctx.config.train_cfg` over its `CFG` for a whitelist of keys —
extend the whitelist if you need to make a new key configurable.

## 6. Datasets

Auto-downloadable:
- `synthetic_smoke`, `synthetic_imbalanced` — generated in `output/data/`, no network
- `fer2013` — Kaggle (`KAGGLE_USERNAME` + `KAGGLE_KEY` in `.env`, or `~/.kaggle/kaggle.json`)

Local-only (set env vars):
- `empath` — `EMPATH_AFFECTNET_DIR`, `EMPATH_RAFDB_DIR`, `EMPATH_SFEW_DIR`
  + optional `EMPATH_FACE_CROP=1` to run the notebook 1 YOLO crop step in-line
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

## 8. Test your deployed model in isolation

Run `make dev`, then open <http://localhost:5173/emotion-test/>.

It's a bare harness — just webcam → face crop → emotion classifier.
No chat, no LLM, no transcripts. You can:
- Watch the live emotion + confidence per frame
- Preview the face crop the model is actually seeing
- Flip `force_label` / `cycle_test_labels` / `log_predictions` from the
  sidebar to debug the path without restarting the service
- Track running prediction counts per class

The full app at `/` still works as normal.

## 9. Adding a new model (not from a notebook)

Mirror an existing folder:

```
pipeline/models/<your_model>/
  __init__.py     # exports build(num_classes) + train(ctx, dataset)
  model.py        # nn.Module definition
  augment.py      # TRAIN_TF / VAL_TF
  train_loop.py   # CFG + run(ctx, dataset, model)
```

Wire it into `pipeline/train.py::RUNS`. If your training procedure is
generic (CE + AdamW), use `pipeline.training.standard.train_classifier`
instead of writing a custom loop — see `pipeline/models/mlp/__init__.py`
for the minimal pattern.

End with a call to `pipeline.training.reporting.write_standard_artifacts`
so your model emits the same artifact shape as everyone else's.

## Questions

- Audit of port fidelity: PR #21 description + the in-conversation audit
- Pipeline architecture: `pipeline/framework/{store,config,context,specs}.py`
  + their docstrings
- How a run dir is structured: this doc § 4
- What live debug flags exist: `application/model_service/core/debug_flags.py`
