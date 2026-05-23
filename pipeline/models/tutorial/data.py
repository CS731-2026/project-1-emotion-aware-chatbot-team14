"""Dataset class — CSV (`path,label`) → batches of (image, label).

Copy verbatim for the common case. See pipeline/models/empathbot_v1/data.py
for per-sample routing (stronger augment for hard classes, etc).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CsvImageDataset(Dataset):
    def __init__(self, csv_path, transform) -> None:
        df = pd.read_csv(csv_path)
        # Drop missing files at init so __getitem__ can trust the path.
        valid = df["path"].apply(lambda p: Path(p).exists())
        self.df = df[valid].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Force RGB — some FER sets ship grayscale.
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), int(row["label"])
