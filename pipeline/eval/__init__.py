"""Evaluation phase + shared eval helpers.

Every trained model gets fed through the same evaluation: same test
sets, same metrics, same artifacts. Apples-to-apples comparison falls
out of that — `make compare` then ranks runs against the same yardstick.

Public surface:
  compute_eval(model, loader, device, num_classes, class_names) -> metrics dict
  write_eval_artifacts(metrics, out_dir)
  load_eval_dataset(name, ctx) -> DatasetSpec (test-only)
  evaluate_baseline(model_id, repo_root, datasets) -> path to eval dir

Internal layout:
  metrics.py      pure metric computation (sklearn)
  artifacts.py    JSON + PNG writers
  loader.py       dynamic import of an eval-named dataset module
  baselines.py    CLI for evaluating already-trained checkpoints
                  (e.g. models/empathbot/empath_final.pth) without a run dir
"""

from .artifacts import write_eval_artifacts
from .loader import load_eval_dataset
from .metrics import compute_eval

__all__ = ["compute_eval", "write_eval_artifacts", "load_eval_dataset"]
