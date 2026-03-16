"""Pre-flight pipeline tests: MLP classifier contracts.

No GPU required. Tests shapes, contracts, and differentiability — not accuracy.
"""
import sys
from pathlib import Path

# Ensure example/ is on the path when run via pytest directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from pipeline import MLP, classification_loss, accuracy

INPUT_DIM = 8
HIDDEN_DIM = 16
NUM_CLASSES = 3
BATCH = 4


def test_mlp_output_shape():
    model = MLP(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)
    x = torch.randn(BATCH, INPUT_DIM)
    logits = model(x)
    assert logits.shape == (BATCH, NUM_CLASSES), f"expected ({BATCH}, {NUM_CLASSES}), got {logits.shape}"


def test_loss_returns_scalar():
    logits = torch.randn(BATCH, NUM_CLASSES)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))
    loss = classification_loss(logits, labels)
    assert loss.shape == (), f"expected scalar, got {loss.shape}"
    assert torch.isfinite(loss), "loss is not finite"


def test_loss_is_differentiable():
    model = MLP(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)
    x = torch.randn(BATCH, INPUT_DIM)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))
    loss = classification_loss(model(x), labels)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"


def test_accuracy_range():
    logits = torch.randn(BATCH, NUM_CLASSES)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))
    acc = accuracy(logits, labels)
    assert 0.0 <= acc <= 1.0, f"accuracy out of range: {acc}"


def test_accuracy_perfect():
    # logits with large values in the correct class → perfect accuracy
    labels = torch.tensor([0, 1, 2, 0])
    logits = torch.zeros(4, NUM_CLASSES)
    logits[0, 0] = 10.0
    logits[1, 1] = 10.0
    logits[2, 2] = 10.0
    logits[3, 0] = 10.0
    assert accuracy(logits, labels) == 1.0
