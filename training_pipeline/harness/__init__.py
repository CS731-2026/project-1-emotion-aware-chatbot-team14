"""Harness — public API."""
import json
from pathlib import Path

from .artifacts import FolderSet, make_folders
from .log_channels import LogChannels, make_log_channels
from .run import run
from .serialization import SerializationError, register_serializer
from .store import Store
from .types import Failure, Loop, Success

__all__ = [
    "loop",
    "main",
    "run",
    "Loop",
    "Success",
    "Failure",
    "Store",
    "make_folders",
    "FolderSet",
    "make_log_channels",
    "LogChannels",
    "register_serializer",
    "SerializationError",
]


def loop(*steps, n, iter_name: str = "iter", while_=None) -> Loop:
    """Create a Loop compound step for use in harness.run().

    Args:
        *steps:    Child steps (callables or nested Loop objects).
        n:         Maximum iterations. Int or callable (config) -> int.
        iter_name: Name used for folder names and the store key injected
                   before each iteration (e.g. "epoch" → epoch_000/,
                   store["epoch"] = 0).
        while_:    Optional continue predicate (store, config) -> bool,
                   evaluated after each completed iteration. Loop exits
                   early if it returns False.

    Example::

        harness.loop(
            train_epoch,
            validate,
            save_best_checkpoint,
            n=lambda config: config["epochs"],
            iter_name="epoch",
            while_=lambda store, config: not store.get("early_stop", False),
        )
    """
    return Loop(steps=list(steps), n=n, iter_name=iter_name, while_=while_)


def main(routines: dict) -> None:
    """Parse CLI args and invoke the named routine.

    Usage in your entry point::

        if __name__ == "__main__":
            harness.main(routines={"train": train_fn})

    CLI::

        python run.py --routine train --config base.yaml --config debug.yaml
    """
    import argparse
    import sys

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--routine", required=True, metavar="NAME",
                        help="Routine to run.")
    parser.add_argument("--config", action="append", default=[],
                        metavar="FILE", dest="configs",
                        help="Config YAML file (repeatable; merged left-to-right).")
    args = parser.parse_args()

    if args.routine not in routines:
        available = sorted(routines.keys())
        print(f"error: unknown routine {args.routine!r}. Available: {available}",
              file=sys.stderr)
        sys.exit(1)

    config: dict = {}
    for path in args.configs:
        with open(path) as f:
            layer = yaml.safe_load(f) or {}
        config = _deep_merge(config, layer)

    routines[args.routine](config)


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Built-in serialisers for infrastructure types.
# These use unique suffixes so they never collide with user data.
# ---------------------------------------------------------------------------

def _save_folder_set(path: Path, obj: FolderSet) -> None:
    path.with_suffix(".folderset").write_text(json.dumps(obj.to_dict(), indent=2))


def _load_folder_set(path: Path) -> FolderSet:
    data = json.loads(path.with_suffix(".folderset").read_text())
    return FolderSet.from_dict(data)


def _save_log_channels(path: Path, obj: LogChannels) -> None:
    path.with_suffix(".logchannels").write_text(json.dumps(obj.to_dict(), indent=2))


def _load_log_channels(path: Path) -> LogChannels:
    data = json.loads(path.with_suffix(".logchannels").read_text())
    return LogChannels.from_dict(data)


register_serializer(FolderSet, save=_save_folder_set, load=_load_folder_set, suffix=".folderset")
register_serializer(LogChannels, save=_save_log_channels, load=_load_log_channels, suffix=".logchannels")
