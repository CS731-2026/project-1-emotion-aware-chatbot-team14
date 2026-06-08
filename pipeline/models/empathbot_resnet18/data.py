"""EmpathBotDataset, per-class augment routing (same shape as
empathbot_v1's data.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .augment import BASE_AUG, HARD_LABEL_IDS, STRONG_AUG, VAL_TF


class EmpathBotDataset(Dataset):
    def __init__(self, csv_path, hard_ids: Iterable[int] = HARD_LABEL_IDS,
                 is_train: bool = True) -> None:
        df = pd.read_csv(csv_path)
        valid = df["path"].apply(lambda p: Path(p).exists())
        self.df = df[valid].reset_index(drop=True)
        self.hard_ids = set(hard_ids)
        self.is_train = is_train

    def __len__(self) -> int: return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        label = int(row["label"])
        if self.is_train:
            tf = STRONG_AUG if label in self.hard_ids else BASE_AUG
        else:
            tf = VAL_TF
        return tf(img), label
