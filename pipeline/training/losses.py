"""Loss registry. get_loss returns an nn.Module ready to call as
`loss(logits, labels)`. Adding a loss = one new branch + one line in
the supported-names docstring."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


SUPPORTED = ("ce", "ce_label_smooth")


def get_loss(
    name: str,
    class_weights: list[float] | None = None,
    args: dict[str, Any] | None = None,
) -> nn.Module:
    args = args or {}
    weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None

    if name == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    if name == "ce_label_smooth":
        smoothing = float(args.get("label_smoothing", 0.1))
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)

    raise ValueError(f"unknown loss {name!r}. supported: {SUPPORTED}")
