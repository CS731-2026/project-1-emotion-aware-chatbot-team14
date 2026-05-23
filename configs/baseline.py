"""Baseline — sane defaults that should run end-to-end on any
(dataset × model) without tuning."""

NAME = "baseline"

CONFIG = {
    "epochs":      5,
    "batch_size":  64,
    "num_workers": 2,

    "augment":   {"name": "mild"},
    "loss":      {"name": "ce"},
    "optimizer": {"name": "adamw", "args": {"lr": 1.0e-3, "weight_decay": 1.0e-4}},
}
