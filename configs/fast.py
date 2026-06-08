"""Fast, 1 epoch, no aug, small batch. The sweep smoke-test config:
"did I just break the loop for any model/dataset?". Each cell
finishes in seconds on a laptop.

LR is intentionally conservative (1e-4) so heavy pretrained backbones
(EfficientNet-B2, ResNet-18) don't NaN out on small synthetic data
within a single epoch. The thorough config uses a different LR for
real training.
"""

NAME = "fast"

CONFIG = {
    "epochs":      1,
    "batch_size":  32,
    "num_workers": 0,

    "augment":   {"name": "none"},
    "loss":      {"name": "ce"},
    "optimizer": {"name": "adamw", "args": {"lr": 1.0e-4, "weight_decay": 0.0}},
}
