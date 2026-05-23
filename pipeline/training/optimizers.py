"""Optimizer registry. Adding an optimizer = one new branch + one
line in SUPPORTED."""

from __future__ import annotations

from typing import Any, Iterable

import torch


SUPPORTED = ("adamw", "sgd")


def get_optimizer(
    name: str,
    params: Iterable[torch.nn.Parameter],
    args: dict[str, Any] | None = None,
) -> torch.optim.Optimizer:
    args = args or {}
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(args.get("lr", 1e-3)),
            weight_decay=float(args.get("weight_decay", 1e-4)),
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(args.get("lr", 1e-2)),
            momentum=float(args.get("momentum", 0.9)),
            weight_decay=float(args.get("weight_decay", 5e-4)),
        )
    raise ValueError(f"unknown optimizer {name!r}. supported: {SUPPORTED}")
