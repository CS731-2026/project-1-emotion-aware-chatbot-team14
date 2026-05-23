"""Pre-flight pipeline tests: data loading contracts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from harness.store import Store
from pipeline import load_data


def _base_config():
    return {
        "input_dim": 8,
        "num_classes": 3,
        "n_train": 64,
        "n_val": 16,
        "seed": 42,
        "batch_size": 16,
    }


def test_load_data_puts_expected_keys():
    store = Store()
    store.put("run_dir", None)
    outcome = load_data(store, _base_config())
    assert bool(outcome), f"load_data returned {outcome}"
    assert "train_dataset" in store
    assert "val_dataset" in store


def test_dataset_lengths():
    store = Store()
    store.put("run_dir", None)
    config = _base_config()
    load_data(store, config)
    assert len(store.get("train_dataset")) == config["n_train"]
    assert len(store.get("val_dataset")) == config["n_val"]


def test_dataset_item_shapes():
    store = Store()
    store.put("run_dir", None)
    config = _base_config()
    load_data(store, config)
    dataset = store.get("train_dataset")
    x, y = dataset[0]
    assert x.shape == (config["input_dim"],), f"unexpected x shape: {x.shape}"
    assert y.shape == (), f"y should be scalar, got {y.shape}"
    assert y.item() in range(config["num_classes"])
