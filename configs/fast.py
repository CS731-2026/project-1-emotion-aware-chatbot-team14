"""Fast — 1 epoch, no aug, small batch. The sweep smoke-test config:
"did I just break the loop for any model/dataset?". Each cell
finishes in seconds on a laptop.
"""

NAME = "fast"

CONFIG = {
    "epochs":      1,
    "batch_size":  32,
    "num_workers": 0,

    "augment":   {"name": "none"},
    "loss":      {"name": "ce"},
    "optimizer": {"name": "adamw", "args": {"lr": 1.0e-3, "weight_decay": 0.0}},
}
