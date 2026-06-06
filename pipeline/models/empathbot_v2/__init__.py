"""empathbot_v2 — empathbot_final architecture + anti-class-imbalance training.

Identical EfficientNet-B2 architecture to empathbot_final. Only the
training recipe differs, to fix the class-imbalance failures we
observed in the first prod run:

  Per-class F1 from empath__empathbot_final__thorough (2026-05-25):
    confusion     0.92  ← model learned this class well
    trust_relief  0.43
    neutral       0.30
    sadness       0.10
    distrust      0.02  ← effectively never predicted
    fear_anxiety  0.00  ← never predicted at all

The model collapsed onto "confusion + default to neutral/trust_relief".
That's the textbook class-imbalance failure mode. Two CFG changes here
target it directly:

1. ``mixup_alpha: 0.2 → 0.0`` — disable MixUp augmentation.

   MixUp (Zhang et al. 2018, arXiv:1710.09412) linearly interpolates
   pairs of (image, label) and trains on the mixed targets. The paper
   derived its benefit under empirical-risk-minimization assumptions
   that implicitly require balanced classes. On imbalanced data,
   sampling pairs uniformly mixes a rare-class image with a common-class
   image roughly P(common)/P(rare) of the time, systematically biasing
   the model's targets toward the common class for the rare class. This
   is documented in the imbalanced-FER literature (e.g. Wang et al.
   2020, "Suppressing Uncertainties for Large-Scale Facial Expression
   Recognition", CVPR 2020 — uses no MixUp; relies on class-balanced
   losses instead).

2. ``neg_boost: 1.2 → 2.5`` — strengthen the NEGATIVE-class weight.

   The pipeline's compute_class_weights starts from inverse frequency
   (1 / class_count, normalized) then multiplies by ``neg_boost`` for
   the classes listed in NEGATIVE_LABEL_IDS (sadness, fear_anxiety,
   distrust). At 1.2× the boost wasn't strong enough — F1 on those
   three classes averaged 0.04. At 2.5× we're explicitly down-weighting
   the head's incentive to default to neutral/trust_relief when
   uncertain about the negative classes.

   Grounding: Cui et al. 2019 "Class-Balanced Loss Based on Effective
   Number of Samples" (CVPR, arXiv:1901.05555) shows that
   inverse-frequency weighting alone can over-correct on long-tailed
   data; a learned or hand-tuned correction factor (our ``neg_boost``)
   is standard practice for FER datasets with the ~6× class imbalance
   the empath merge produces.

Architecture (model.py, augment.py, data.py) is unchanged from
empathbot_final — re-exports below. The only fresh code is this
docstring's worth of intent + a small ``CFG_OVERRIDES`` dict + a
``train()`` shim that layers the overrides under whatever runs.yaml
passes via train_cfg.
"""

from __future__ import annotations

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel

# Re-exports so the eval phase + tooling find PREPROCESS/build the same
# way they do for empathbot_final.
from pipeline.models.empathbot_final import PREPROCESS
from pipeline.models.empathbot_final.model import build
from pipeline.models.empathbot_final.train_loop import run as _run

__all__ = ["PREPROCESS", "build", "train", "CFG_OVERRIDES"]


# Defaults that differ from empathbot_final. The per-run train_cfg in
# runs.yaml stacks on top of these (user-row wins on conflict).
#
# `mixup_start_epoch: 999` is the actual MixUp disable — the inner
# train_loop gates MixUp via `epoch > mixup_start_epoch`, not via
# `mixup_alpha`. Setting alpha=0 alone would still hit a Beta(0,0)
# crash inside _mixup_batch. Pushing mixup_start_epoch past `epochs`
# (default 25, max practical ~100) bypasses the conditional entirely.
# Keeping mixup_alpha=0.0 too for intent clarity in the hparams log.
CFG_OVERRIDES = {
    "mixup_alpha":       0.0,
    "mixup_start_epoch": 999,
    "neg_boost":         2.5,
}


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    """Layer CFG_OVERRIDES under the row's train_cfg, then delegate to
    empathbot_final's inner training loop. Identical training procedure,
    different hyperparameter defaults."""
    # ctx.config is frozen but train_cfg (the dict inside) is mutable.
    # Apply our overrides BENEATH the row's explicit train_cfg so a
    # runs.yaml row can still override them if needed.
    merged = {**CFG_OVERRIDES, **ctx.config.train_cfg}
    ctx.config.train_cfg.clear()
    ctx.config.train_cfg.update(merged)
    return _run(ctx, dataset, model=build(dataset.num_classes))
