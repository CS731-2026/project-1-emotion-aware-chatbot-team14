"""data.py — Dataset class (how the CSV becomes batches).

The dataset module (pipeline/datasets/<name>/) produces three CSVs
per split (train.csv, val.csv, test.csv) each with at least:
    path,label
This file converts a CSV path into a PyTorch Dataset that yields
(transformed_image, int_label) pairs.

For the common case — a simple CSV with no special routing — you can
copy this verbatim. The interesting per-sample-routing patterns
(stronger augment for hard classes, weighted sampling, blur filters,
etc.) live in pipeline/models/empathbot_v1/data.py if you want to see
a richer example.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CsvImageDataset(Dataset):
    """Reads a CSV of (path, label) rows and yields (image, label) tuples.

    Why filter for `Path(p).exists()` at init?
      Datasets sometimes ship with broken rows (the dataset module
      logs a warning when it sees these). Filtering at init means
      __getitem__ is allowed to assume the path is valid — no
      surprise FileNotFoundErrors mid-epoch.

    Why `Image.open(...).convert("RGB")`?
      Some FER datasets ship grayscale, some RGB, some RGBA. Forcing
      RGB normalises and lets the same transforms work for all.

    Why a per-instance `transform`?
      So train and val can use different transforms (augmented vs
      deterministic) while sharing the dataset class. See augment.py
      for the two transform definitions.
    """

    def __init__(self, csv_path, transform) -> None:
        df = pd.read_csv(csv_path)
        valid = df["path"].apply(lambda p: Path(p).exists())
        self.df = df[valid].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), int(row["label"])
