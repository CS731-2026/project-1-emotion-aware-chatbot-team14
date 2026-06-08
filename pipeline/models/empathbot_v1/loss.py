"""FocalLoss, lifted verbatim from notebook 6b cell 11.

Focal Loss (Lin et al. 2017) with class weighting and label smoothing.
  gamma=0 → standard CrossEntropy
  gamma=2 → focuses 4x more on samples the model gets wrong

The notebook is the source of truth; any tweak should land there first
and be re-lifted here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor, gamma: float = 2.0,
                 label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = int(weight.numel())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            smooth = self.label_smoothing / (self.num_classes - 1)
            one_hot = torch.full_like(logits, smooth)
            one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()

        # Focal weight: (1 - p_t)^gamma
        p_t = (prob * one_hot).sum(dim=1)
        focal_w = (1.0 - p_t) ** self.gamma

        # Class weight for each sample
        weight = self.weight if isinstance(self.weight, torch.Tensor) else torch.as_tensor(self.weight)
        class_w = weight[targets]

        loss = -(one_hot * log_prob).sum(dim=1)
        loss = focal_w * class_w * loss
        return loss.mean()
