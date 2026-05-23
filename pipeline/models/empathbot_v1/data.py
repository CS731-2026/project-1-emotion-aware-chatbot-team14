"""EmpathBotDataset + per-class augmentation routing.

Ported from notebook 6b cell 7. Selects STRONG_AUG for "hard" classes
(sadness, fear_anxiety, distrust per HARD_LABEL_IDS) and BASE_AUG for
the rest. Val/test use VAL_TF bare.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .augment import BASE_AUG, HARD_LABEL_IDS, STRONG_AUG, VAL_TF


class EmpathBotDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        hard_ids: Iterable[int] = HARD_LABEL_IDS,
        is_train: bool = True,
    ) -> None:
        df = pd.read_csv(csv_path)
        valid = df["path"].apply(lambda p: Path(p).exists())
        self.df = df[valid].reset_index(drop=True)
        self.hard_ids = set(hard_ids)
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        label = int(row["label"])
        if self.is_train:
            tf = STRONG_AUG if label in self.hard_ids else BASE_AUG
        else:
            tf = VAL_TF
        return tf(img), label
