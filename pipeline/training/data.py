"""CSV-driven image dataset shared across every split + every model.

Reads a CSV with columns ('path', 'label'), loads the image with PIL,
runs it through whatever transform the caller passes (augment + model
PREPROCESS for train; PREPROCESS only for val/test).

PIL's .convert("RGB") handles grayscale-on-disk transparently, FER2013
ships 1-channel images but our models all expect 3, and replicating
the channel via PIL is cleaner than a torch transform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CsvImageDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform: Callable[[Any], Any] | None = None) -> None:
        self._df = pd.read_csv(csv_path)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):
        row = self._df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self._transform is not None:
            img = self._transform(img)
        return img, int(row["label"])


def make_loader(
    csv_path: str | Path,
    transform: Callable[[Any], Any] | None,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        CsvImageDataset(csv_path, transform=transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
