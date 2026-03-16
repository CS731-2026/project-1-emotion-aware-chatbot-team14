"""Harness tests: Store contracts and serialiser round-trips.

These test the harness infrastructure itself — not any particular experiment.
Run during harness development: python -m pytest harness_tests/ -v
"""
import tempfile
from pathlib import Path

import torch
import pytest

from harness.store import Store
from harness.serialization import load_value, save_value, SerializationError


def test_store_get_missing_raises():
    store = Store()
    with pytest.raises(KeyError, match="not found"):
        store.get("missing_key")


def test_store_put_and_get():
    store = Store()
    store.put("x", 42)
    assert store.get("x") == 42


def test_store_overwrite_is_silent():
    store = Store()
    store.put("x", 1)
    store.put("x", 2)
    assert store.get("x") == 2


def test_store_clear():
    store = Store()
    store.put("x", 1)
    store.clear()
    with pytest.raises(KeyError):
        store.get("x")


def test_store_contains():
    store = Store()
    store.put("x", 1)
    assert "x" in store
    assert "y" not in store


def test_serialiser_tensor_roundtrip():
    tensor = torch.randn(4, 8)
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "tensor"
        save_value(stem, tensor)
        loaded = load_value(stem)
    assert torch.allclose(tensor, loaded)


def test_serialiser_json_roundtrip():
    data = {"lr": 0.001, "epochs": 10, "tags": ["a", "b"]}
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "config"
        save_value(stem, data)
        loaded = load_value(stem)
    assert loaded == data


def test_serialiser_pickle_roundtrip():
    from torch.utils.data import TensorDataset
    ds = TensorDataset(torch.randn(8, 4), torch.zeros(8, dtype=torch.long))
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "dataset"
        save_value(stem, ds)
        loaded = load_value(stem)
    x_orig, _ = ds[0]
    x_load, _ = loaded[0]
    assert torch.allclose(x_orig, x_load)


def test_serialiser_nn_module_roundtrip():
    import torch.nn as nn
    model = nn.Linear(4, 2)
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "model"
        save_value(stem, model)
        loaded = load_value(stem)
    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), loaded.state_dict().items()
    ):
        assert k1 == k2
        assert torch.allclose(v1, v2)


def test_folderset_roundtrip():
    """FolderSet survives serialise → deserialise cycle."""
    import json
    from harness.artifacts import FolderSet

    # Import triggers the register_serializer calls
    import harness  # noqa: F401

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        folders = FolderSet({"checkpoints": tmp / "ckpt", "eval": tmp / "eval"})
        stem = tmp / "folders"
        save_value(stem, folders)
        loaded = load_value(stem)

    assert set(loaded._folders.keys()) == {"checkpoints", "eval"}


def test_logchannels_roundtrip():
    """LogChannels survives serialise → deserialise cycle."""
    from harness.log_channels import LogChannels, LogChannel
    import harness  # noqa: F401 — triggers registrations

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ch = LogChannels({"training": LogChannel(tmp / "training.log")})
        ch.training("hello")
        stem = tmp / "logs"
        save_value(stem, ch)
        loaded = load_value(stem)
        loaded.training("world")
        assert (tmp / "training.log").exists()
