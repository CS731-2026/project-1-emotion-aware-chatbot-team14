from __future__ import annotations

from pathlib import Path
from typing import Any

from .serialization import load_value, save_value


class FolderSet:
    """Named output folders for pipeline-defined, public-facing artifacts.

    Created once by a setup step via make_folders() and stored in the store.
    Steps write and read explicit artifacts here — model checkpoints, evaluation
    results, summaries — distinct from the per-step store snapshots run() manages.
    """

    def __init__(self, folders: dict[str, Path]) -> None:
        self._folders = folders

    def save(self, folder: str, filename: str, value: Any) -> Path:
        """Serialise value into the named folder.

        Args:
            folder: Key from the folders dict (e.g. "checkpoints").
            filename: Stem for the output file (suffix appended by serialiser).
            value: Value to serialise.

        Returns:
            Path to the written file (with suffix).
        """
        dest_dir = self._resolve(folder)
        path_stem = dest_dir / filename
        suffix = save_value(path_stem, value)
        return path_stem.with_suffix(suffix)

    def load(self, folder: str, filename: str) -> Any:
        """Load a previously saved artifact from the named folder.

        Args:
            folder: Key from the folders dict.
            filename: Stem of the file (without suffix).
        """
        dest_dir = self._resolve(folder)
        return load_value(dest_dir / filename)

    def path(self, folder: str) -> Path:
        """Return the directory path for a named folder."""
        return self._resolve(folder)

    def _resolve(self, folder: str) -> Path:
        if folder not in self._folders:
            raise KeyError(
                f"folder {folder!r} not defined. "
                f"Available folders: {sorted(self._folders.keys())}"
            )
        return self._folders[folder]

    # --- Serialisation support ---

    def to_dict(self) -> dict:
        return {name: str(p) for name, p in self._folders.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "FolderSet":
        return cls({name: Path(p) for name, p in data.items()})


def make_folders(run_dir: Path, folders: dict[str, str]) -> FolderSet:
    """Create named output folders under run_dir.

    Args:
        run_dir: Root of the current run directory.
        folders: Mapping of logical name -> relative path under run_dir.
                 e.g. {"checkpoints": "checkpoints", "eval": "eval"}

    Returns:
        FolderSet instance.
    """
    built: dict[str, Path] = {}
    for name, rel_path in folders.items():
        p = run_dir / rel_path
        p.mkdir(parents=True, exist_ok=True)
        built[name] = p
    return FolderSet(built)
