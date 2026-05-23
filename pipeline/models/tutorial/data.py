"""Dataset class — turns the CSV (`path,label`) into (image, label) batches.

Common case; copy verbatim. For per-sample routing (e.g. stronger augment
for hard classes, weighted sampling, blur filters) see
pipeline/models/empathbot_v1/data.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CsvImageDataset(Dataset):
    def __init__(self, csv_path, transform) -> None:
        df = pd.read_csv(csv_path)

        # Drop rows whose files no longer exist (dataset module logs a warning
        # when it sees these). After this, __getitem__ can trust the path.
        valid = df["path"].apply(lambda p: Path(p).exists())
        self.df = df[valid].reset_index(drop=True)
        self.transform = transform                # train vs val transform per-loader

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")  # force RGB (some FER sets ship grayscale)
        return self.transform(img), int(row["label"])
