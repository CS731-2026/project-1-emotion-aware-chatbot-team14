"""EmpathBot dataset with NEGATIVE-class-aware augmentation routing.

Ported from notebook 5_v4 cell 14. Reads (path, label) from a CSV
and applies NEG_TF to NEGATIVE classes, STD_TF to the rest. Val/test
use VAL_TF bare.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .augment import NEG_TF, NEGATIVE_LABEL_IDS, STD_TF, VAL_TF


class EmpathBotDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        is_train: bool,
        negative_ids: Iterable[int] = NEGATIVE_LABEL_IDS,
    ) -> None:
        df = pd.read_csv(csv_path)
        valid = df["path"].apply(lambda p: Path(p).exists())
        self.df = df[valid].reset_index(drop=True)
        self.is_train = is_train
        self.negative_ids = set(negative_ids)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        label = int(row["label"])
        if self.is_train:
            tf = NEG_TF if label in self.negative_ids else STD_TF
        else:
            tf = VAL_TF
        return tf(img), label

    def class_counts(self, num_classes: int) -> list[int]:
        """Per-class sample count — used by compute_class_weights."""
        counts = [0] * num_classes
        for _, row in self.df.iterrows():
            label = int(row["label"])
            if 0 <= label < num_classes:
                counts[label] += 1
        return counts
