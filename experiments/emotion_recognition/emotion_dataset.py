"""
CS731 — PyTorch Dataset Class
==============================
Loads images from a split manifest CSV produced by dataset_preparation.py.
Supports train augmentation and val/test plain transforms.

Usage
-----
  from data.emotion_dataset import EmotionDataset, get_transforms, get_loaders
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: str = 'train', img_size: int = 224) -> transforms.Compose:
    """
    Returns the appropriate transform pipeline for each split.

    Train: aggressive augmentation (matches both exemplar teams).
    Val / Test: only resize + normalise (no augmentation — avoids data leakage).
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size + 16, img_size + 16)),   # slightly larger
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                    scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class EmotionDataset(Dataset):
    """
    Loads (image, label_id) pairs from a split CSV manifest.

    CSV columns required: path, label, label_id
    """

    def __init__(self, csv_path: str | Path, transform=None):
        self.df = pd.read_csv(csv_path)
        assert 'path'     in self.df.columns, 'CSV missing "path" column'
        assert 'label'    in self.df.columns, 'CSV missing "label" column'
        assert 'label_id' in self.df.columns, 'CSV missing "label_id" column'

        # Drop rows with missing files (safety check)
        valid = self.df['path'].apply(lambda p: Path(p).exists())
        n_missing = (~valid).sum()
        if n_missing > 0:
            print(f'[WARN] EmotionDataset: dropping {n_missing} missing files.')
        self.df = self.df[valid].reset_index(drop=True)

        self.transform = transform
        self.classes   = sorted(self.df['label'].unique())
        self.num_classes = len(self.classes)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def class_weights(self) -> torch.Tensor:
        """
        Returns per-sample weights for WeightedRandomSampler.
        Upsamples minority classes to address class imbalance.
        """
        counts = self.df['label_id'].value_counts().sort_index()
        freq   = counts / counts.sum()
        weight_per_class = 1.0 / freq
        sample_weights   = self.df['label_id'].map(weight_per_class).values
        return torch.tensor(sample_weights, dtype=torch.float)

    # ── Standard Dataset interface ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = int(row['label_id'])
        return image, label

    def __repr__(self) -> str:
        return (f'EmotionDataset(n={len(self)}, '
                f'classes={self.classes})')


# ── MixUp & CutMix ───────────────────────────────────────────────────────────

def mixup_data(x: torch.Tensor, y: torch.Tensor,
               alpha: float = 0.4) -> tuple:
    """
    MixUp augmentation (Zhang et al., 2018).
    Linearly interpolates image pairs and their labels.

    Returns: mixed_x, y_a, y_b, lambda
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Computes MixUp loss as weighted sum of two CE losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x: torch.Tensor, y: torch.Tensor,
                alpha: float = 1.0) -> tuple:
    """
    CutMix augmentation (Yun et al., 2019).
    Cuts and pastes random rectangular regions between images.

    Returns: mixed_x, y_a, y_b, lambda
    """
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size, device=x.device)

    cut_rat = np.sqrt(1 - lam)
    cut_w   = int(W * cut_rat)
    cut_h   = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (W * H)
    return mixed_x, y, y[index], lam


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_loaders(
    splits_dir:  str | Path,
    mode:        str = 'ekman6',
    batch_size:  int = 32,
    img_size:    int = 224,
    num_workers: int = 4,
    use_sampler: bool = True,   # WeightedRandomSampler for class imbalance
) -> dict:
    """
    Build DataLoaders for all available splits.

    Args:
        splits_dir:  directory containing <mode>_train.csv, <mode>_val.csv, etc.
        mode:        'ekman6' | 'ekman7' | 'all8'
        batch_size:  mini-batch size
        img_size:    input resolution (224 for most timm models)
        num_workers: parallel data loading workers
        use_sampler: balance batches via WeightedRandomSampler (train only)

    Returns:
        {'train': DataLoader, 'val': DataLoader, 'test': DataLoader (optional)}
    """
    splits_dir = Path(splits_dir)
    loaders    = {}

    for split in ('train', 'val', 'test'):
        csv_path = splits_dir / f'{mode}_{split}.csv'
        if not csv_path.exists():
            continue  # test split is optional

        transform = get_transforms(split, img_size)
        dataset   = EmotionDataset(csv_path, transform=transform)

        if split == 'train' and use_sampler:
            sampler = WeightedRandomSampler(
                weights     = dataset.class_weights,
                num_samples = len(dataset),
                replacement = True
            )
            loader = DataLoader(
                dataset,
                batch_size  = batch_size,
                sampler     = sampler,
                num_workers = num_workers,
                pin_memory  = True,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size  = batch_size,
                shuffle     = False,
                num_workers = num_workers,
                pin_memory  = True,
            )

        loaders[split] = loader
        print(f'[INFO] {split:6s} loader: {len(dataset):,} images, '
              f'{len(loader)} batches, classes={dataset.classes}')

    return loaders


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python data/emotion_dataset.py <splits_dir> [mode]')
        print('Example: python data/emotion_dataset.py data/splits ekman6')
        sys.exit(0)

    splits_dir = sys.argv[1]
    mode       = sys.argv[2] if len(sys.argv) > 2 else 'ekman6'
    loaders    = get_loaders(splits_dir, mode, batch_size=8, num_workers=0)

    for name, loader in loaders.items():
        imgs, labels = next(iter(loader))
        print(f'{name}: images={imgs.shape}, labels={labels}')
