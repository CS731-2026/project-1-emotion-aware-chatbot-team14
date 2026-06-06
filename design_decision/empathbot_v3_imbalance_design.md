# EmpathBot v3+ — Class-Imbalance Design Plan

**Status:** design only, no code yet.
**Authors:** TJ + Claude (pair-design session 2026-05-25)
**Context:** prod runs v1 and v2 surfaced a class-collapse failure mode that needs to be addressed before further architecture work.

---

## What we've established so far

| Run | Config | Outcome | Diagnosis |
|---|---|---|---|
| **v1** (`empathbot_final`) | MixUp@0.2, neg_boost=1.2, batch=64, 25 epochs | acc 35.3%, macro F1 0.29, F1=**0** on fear_anxiety, F1=**0.02** on distrust | Classic class-collapse — model learned `confusion` (F1=0.92) then defaulted to `neutral`/`trust_relief` for everything else |
| **v2** (`empathbot_v2`) | MixUp off, neg_boost=2.5, batch=64, 25 epochs (killed @ ep 8) | train_acc 84%, val_acc 46% (38pp gap) | Severe overfit — MixUp was doing meaningful regularization work that we removed |

**Inferences carried forward:**
1. MixUp is genuinely useful as a regularizer for empathbot_final on empath
2. MixUp on its own amplifies class-imbalance (rare-class samples get systematically blended with common-class samples)
3. `neg_boost` (multiplicative loss weight) at 2.5× is not enough to overcome the imbalance
4. **No single intervention solves both regularization and imbalance** — we need a layered fix

---

## Class-imbalance toolkit (by intervention layer)

| Layer | Technique | What it does | Effort | Evidence |
|---|---|---|---|---|
| **Sampling** | `WeightedRandomSampler` | Oversample rare classes — each epoch sees ~equal counts per class | Small | Universal practice for severe imbalance |
| **Data aug** | Class-aware MixUp | Mix only within-class samples, OR mix at low alpha so soft labels stay near one-hot | Small | Wang et al. 2020 CVPR (FER-specific) |
| **Data aug** | Stronger augmentation on rare classes | Crank up `NEG_TF` (rotation, color jitter, cutout) specifically for sad/fear/distrust | Trivial | Standard FER practice |
| **Loss** | Focal Loss (Lin et al. 2017, [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)) | `-(1-p_t)^γ · log(p_t)` — downweights confident predictions, forces continued focus on hard examples | Small | SOTA on object detection imbalance; widely used in FER |
| **Loss** | Class-balanced loss (Cui et al. 2019, [arXiv:1901.05555](https://arxiv.org/abs/1901.05555)) | Replaces inverse-freq with "effective number of samples" — gentler on extremely rare classes | Small | More principled than naive inverse-freq |
| **Loss** | LDAM (Cao et al. 2019, [arXiv:1906.07413](https://arxiv.org/abs/1906.07413)) | Margin-aware loss that pushes decision boundary further from minority class samples | Medium | Strong empirical results on long-tail benchmarks |
| **Schedule** | Deferred reweighting (DRW, Cao 2019) | Train without weights first (good feature learning), enable weights in second half (boundary correction) | Medium | Often beats from-epoch-1 weighting |
| **Architecture** | Per-class attention heads | Separate "fear specialist" / "distrust specialist" sub-heads on shared features | Large | DACL, DAN family of FER architectures |

---

## Proposed **v3** — focused, layered, leaves room for v4 architecture changes

Three concurrent interventions, each targeting a different layer. Bundled to address imbalance comprehensively while restoring regularization.

### 1. Re-enable MixUp at reduced alpha *(data augmentation)*
- `mixup_alpha: 0.1` (was 0.2 in v1, 0.0 in v2)
- Cuts cross-class mixing rate in half vs v1
- Keeps regularization benefit (v2 proved we need it)
- Soft labels stay closer to one-hot → rare-class gradient signal less diluted

### 2. Add `WeightedRandomSampler` to DataLoader *(data sampling — most important)*
- Sample probabilities ∝ inverse class frequency
- Each epoch sees ~equal samples per class regardless of underlying distribution
- This is the **structural fix** to imbalance. Loss weights are a hint; sampler enforcement is concrete.
- Lean toward **square-root weighting** rather than full inverse-frequency (more stable, less variance)

### 3. Replace weighted CE with Focal Loss *(loss)*
- `gamma=2.0` (Lin et al. 2017 paper default)
- Combined with sampler oversampling, focal loss keeps pushing on the hardest examples even after easy ones are solved
- New addition to `pipeline/training/losses.py` (~15-line function)

**Predicted outcome:** F1 on fear_anxiety and distrust climbs from 0.00 / 0.02 → likely **0.30-0.45 range**. Aggregate accuracy might slightly dip but macro F1 should climb meaningfully (current 0.29 → likely 0.40-0.50).

### v3 code footprint

| Change | File | Lines |
|---|---|---|
| New module `pipeline/models/empathbot_balanced/` | new | ~50 (docstring + imports + thin wrapper) |
| New `train_loop.py` with sampler + focal loss | new | ~150 (mostly copy of empathbot_final's `run()` with the two swaps) |
| `FocalLoss` class | `pipeline/training/losses.py` | ~15 |
| Row in `runs.yaml` | edit | 5 |
| **Total** | | **~200 lines**, ~1 hour to write |

**Naming:** call the new module `empathbot_balanced` (descriptive of intent). The existing `empathbot_v3` is a notebook port and would conflict.

---

## v4+ — architecture iteration paths *(after v3 stabilizes)*

Three orthogonal architecture experiments worth queuing once v3 gives us a properly-trained baseline:

### v4a — bigger backbone, same recipe
- `backbone: efficientnet_b3` (12M params) or `b4` (19M params, requires bumping image size to 380)
- Trivial change — `EmpathBotV1` is parameterized on backbone
- Cost: ~1.5–2× longer training per epoch
- Expected: +2-5pp accuracy if not data-limited

### v4b — POSTER++ *(already vendored!)*
- `pipeline/models/posterplus/` exists, currently commented out in `runs.yaml`
- Mao et al. 2023 ([arXiv:2301.12149](https://arxiv.org/abs/2301.12149)) — SOTA on RAF-DB at publication
- Two-stream architecture (face image + landmark coords) with cross-attention
- Cost: more setup (`make install-training` activates vendored POSTER_V2)
- Expected: potentially significant jump on minority classes (designed for FER)

### v4c — ConvNeXt-Base or Swin-Tiny
- Modern alternatives to EfficientNet
- ConvNeXt: pure conv with transformer-inspired design (Liu et al. 2022)
- Swin-Tiny: hierarchical transformer, ~28M params, captures global facial relationships
- Cost: new model module per backbone (~100 lines each)
- Expected: modest accuracy gain at cost of training time

---

## v5+ — multi-dataset / multi-task *(longer term)*

Worth considering once architecture work stabilizes:

- **Pretrain on FER2013, fine-tune on empath** — large-scale pretraining then domain transfer
- **Multi-task: emotion + AU prediction** — auxiliary head predicts Facial Action Units (FACS), forces backbone to learn facial-muscle features. Datasets like BP4D include AU labels.
- **Combine FER2013 + empath into one train set** with dataset-id auxiliary input — the model sees both distributions and learns domain-invariant features

---

## Open design questions

1. **Are these three v3 interventions the right bundle?** Or vary one at a time (slower but better attribution)?
2. **Focal Loss gamma value** — 2.0 paper default, vs 1.0 (gentler) or 3.0 (more aggressive)
3. **WeightedRandomSampler strength** — full inverse frequency (each class exactly equal) or square-root partial (more stable)?
4. **Enable backbone unfreeze earlier?** (epoch 3 vs 5). With sampler oversampling, the head has more diverse data in fewer epochs — earlier unfreeze might compound benefits.
5. **For v4: cheap bigger-backbone (v4a) first, or jump to SOTA POSTER++ (v4b)?**
6. **Multi-seed runs for v3?** Seed variance is ±2pp. Single-seed v3 vs v1 comparison could be misleading. Worth running v3 with 2-3 seeds before declaring success?

---

## What v3 needs in code (specific paths)

```
pipeline/
├── models/
│   └── empathbot_balanced/             ← NEW
│       ├── __init__.py                  thin wrapper + CFG + train() shim
│       └── train_loop.py                copy of empathbot_final/train_loop.py
│                                          with WeightedRandomSampler injected,
│                                          loss swapped for FocalLoss
├── training/
│   └── losses.py                        ← EDIT — add FocalLoss class
runs.yaml                                ← EDIT — add empath x empathbot_balanced row
```

The new module follows the team's "new files only" rule. No existing model files modified.

---

## Decision log

| Date | Choice | Rationale |
|---|---|---|
| 2026-05-25 | Saved this design without implementing | TJ wanted to capture state before context-switching to the system diagram work |
| 2026-05-25 | Picked `empathbot_balanced` as module name | `empathbot_v3` already taken by an existing notebook port |
| 2026-05-25 | Three-intervention v3 (not one-at-a-time) | Time-constrained; can do attribution ablations later if v3 results are unclear |

---

## References (academic grounding for each technique)

- **MixUp** — Zhang et al. 2018 "mixup: Beyond Empirical Risk Minimization" [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
- **Focal Loss** — Lin et al. 2017 "Focal Loss for Dense Object Detection" [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- **Class-Balanced Loss** — Cui et al. 2019 "Class-Balanced Loss Based on Effective Number of Samples" [arXiv:1901.05555](https://arxiv.org/abs/1901.05555)
- **LDAM** — Cao et al. 2019 "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss" [arXiv:1906.07413](https://arxiv.org/abs/1906.07413)
- **POSTER++** — Mao et al. 2023 "POSTER V2: A Simpler and Stronger Facial Expression Recognition Network" [arXiv:2301.12149](https://arxiv.org/abs/2301.12149)
- **ConvNeXt** — Liu et al. 2022 "A ConvNet for the 2020s" [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
- **Suppressing Uncertainties for Large-Scale FER** — Wang et al. 2020 CVPR (FER-specific class-balanced training, often cited as MixUp-disable precedent)
