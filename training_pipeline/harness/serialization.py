from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Callable


class SerializationError(Exception):
    pass


# Registry: list of (type, suffix, save_fn, load_fn) tried in order.
# First match wins.
_REGISTRY: list[tuple[type, str, Callable, Callable]] = []


def register_serializer(
    type: type,
    save: Callable[[Path, Any], None],
    load: Callable[[Path], Any],
    suffix: str = ".pkl",
) -> None:
    """Register a custom serialiser for a type.

    Registered serialisers are tried before the built-in fallback chain.
    Later registrations for the same type take precedence.

    Args:
        type: The Python type this serialiser handles.
        save: fn(path_without_suffix, value) -> None
        load: fn(path_without_suffix) -> value
        suffix: file suffix the save/load pair uses (default: .pkl)
    """
    # Insert at front so later registrations win
    _REGISTRY.insert(0, (type, suffix, save, load))


def _save_torch(path: Path, value: Any) -> None:
    import torch  # local import — torch is optional at module level

    torch.save(value, path.with_suffix(".pt"))


def _load_torch(path: Path) -> Any:
    import torch

    return torch.load(path.with_suffix(".pt"), weights_only=False)


def _save_numpy(path: Path, value: Any) -> None:
    import numpy as np

    np.save(path.with_suffix(".npy"), value)


def _load_numpy(path: Path) -> Any:
    import numpy as np

    return np.load(path.with_suffix(".npy"), allow_pickle=False)


def _save_json(path: Path, value: Any) -> None:
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(value, f, indent=2)


def _load_json(path: Path) -> Any:
    with open(path.with_suffix(".json")) as f:
        return json.load(f)


def _save_pickle(path: Path, value: Any) -> None:
    with open(path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(value, f)


def _load_pickle(path: Path) -> Any:
    with open(path.with_suffix(".pkl"), "rb") as f:
        return pickle.load(f)


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def save_value(path: Path, value: Any) -> str:
    """Serialise value to disk. Returns the suffix used.

    path is the stem (no suffix). The serialiser appends its own suffix.
    Raises SerializationError if no serialiser can handle the type.
    """
    # 1. Check custom registry first
    for registered_type, suffix, save_fn, _ in _REGISTRY:
        if isinstance(value, registered_type):
            save_fn(path, value)
            return suffix

    # 2. Built-in chain
    try:
        import torch
        import torch.nn as nn

        if isinstance(value, (torch.Tensor, nn.Module)):
            _save_torch(path, value)
            return ".pt"
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            _save_numpy(path, value)
            return ".npy"
    except ImportError:
        pass

    if _is_json_serializable(value):
        _save_json(path, value)
        return ".json"

    # 3. Pickle fallback — try it, but catch non-picklable types cleanly
    try:
        _save_pickle(path, value)
        return ".pkl"
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        type_name = type(value).__name__
        raise SerializationError(
            f"store key at path {path.name!r} holds a value of type {type_name!r}\n"
            f"which cannot be serialised (pickle failed: {exc}).\n\n"
            f"Register a custom serialiser with:\n\n"
            f"    from harness import register_serializer\n"
            f"    register_serializer(\n"
            f"        type={type_name},\n"
            f"        save=lambda path, obj: ...,\n"
            f"        load=lambda path: ...,\n"
            f"    )\n\n"
            f"Or restructure so the step stores a serialisable value instead."
        ) from exc


# Suffix → load function mapping for the built-in chain
_SUFFIX_LOADERS: dict[str, Callable[[Path], Any]] = {
    ".pt": _load_torch,
    ".npy": _load_numpy,
    ".json": _load_json,
    ".pkl": _load_pickle,
}


def load_value(path_stem: Path) -> Any:
    """Load a value from disk given the stem (no suffix).

    Tries custom registry first (by suffix), then built-in loaders.
    Raises FileNotFoundError if no matching file exists.
    """
    # Check custom registry
    for _, suffix, _, load_fn in _REGISTRY:
        candidate = path_stem.with_suffix(suffix)
        if candidate.exists():
            return load_fn(path_stem)

    # Built-in loaders
    for suffix, load_fn in _SUFFIX_LOADERS.items():
        candidate = path_stem.with_suffix(suffix)
        if candidate.exists():
            return load_fn(path_stem)

    raise FileNotFoundError(
        f"No serialised file found for stem {path_stem}. "
        f"Checked suffixes: {list(_SUFFIX_LOADERS.keys())}"
    )
