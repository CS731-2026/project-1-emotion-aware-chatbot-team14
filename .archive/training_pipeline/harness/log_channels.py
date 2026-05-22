from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import IO


class LogChannel:
    """A single named log channel that writes timestamped lines to a file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = open(self._path, "a", buffering=1)  # line-buffered

    def __call__(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._file.write(f"[{ts}] {message}\n")

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()

    @property
    def path(self) -> Path:
        return self._path


class LogChannels:
    """Collection of named log channels, accessible as attributes."""

    def __init__(self, channels: dict[str, LogChannel]) -> None:
        self._channels = channels
        # Expose each channel as an attribute for ergonomic access
        for name, channel in channels.items():
            setattr(self, name, channel)

    def flush_all(self) -> None:
        for ch in self._channels.values():
            ch.flush()

    def close_all(self) -> None:
        for ch in self._channels.values():
            ch.close()

    def snapshot_to(self, dest_dir: Path) -> None:
        """Copy all log files into dest_dir (e.g. a step folder)."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        self.flush_all()
        for ch in self._channels.values():
            if ch.path.exists():
                shutil.copy2(ch.path, dest_dir / ch.path.name)

    def names(self) -> list[str]:
        return list(self._channels.keys())

    # --- Serialisation support ---

    def to_dict(self) -> dict:
        """Return {name: path_str} so the channels can be reconstructed."""
        return {name: str(ch.path) for name, ch in self._channels.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "LogChannels":
        """Reconstruct LogChannels by reopening each log file in append mode."""
        built = {name: LogChannel(Path(p)) for name, p in data.items()}
        return cls(built)


def make_log_channels(run_dir: Path, channels: dict[str, str]) -> LogChannels:
    """Create named log channels writing to run_dir.

    Args:
        run_dir: Root of the current run directory.
        channels: Mapping of channel name -> relative path under run_dir.
                  e.g. {"training": "logs/training.log", "eval": "logs/eval.log"}

    Returns:
        LogChannels instance with each channel accessible as an attribute.
    """
    built: dict[str, LogChannel] = {}
    for name, rel_path in channels.items():
        path = run_dir / rel_path
        built[name] = LogChannel(path)
    return LogChannels(built)
