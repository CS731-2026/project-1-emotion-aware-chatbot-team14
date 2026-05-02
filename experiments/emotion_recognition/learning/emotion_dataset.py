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
    # ImageNet normalization constants — used by all pretrained models from timm
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if split == 'train':
        # TRAINING AUGMENTATION PIPELINE
        # WHY: During training, we apply aggressive transformations to prevent overfitting
        # and improve generalization. The model learns to recognize emotions across
        # different poses, lighting, and image qualities.
        return transforms.Compose([
            # Step 1: Resize to slightly larger (224+16) before cropping
            # WHY: Allows random crop to still hit the subject even at edges
            transforms.Resize((img_size + 16, img_size + 16)),

            # Step 2: Randomly crop to target size with zoom variation (0.8x to 1.0x)
            # DATATYPE: Image tensor changes from (C, 240, 240) to (C, 224, 224)
            # WHY: Simulates camera zoom/distance variation; encourages learning at different scales
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),

            # Step 3: Random horizontal flip (50% probability)
            # WHY: Emotions are generally symmetric; doubles effective dataset size
            transforms.RandomHorizontalFlip(p=0.5),

            # Step 4: Random rotation (±15 degrees)
            # WHY: Head tilt variation; common in real-world video
            transforms.RandomRotation(degrees=15),

            # Step 5: Color jitter (brightness, contrast, saturation, hue)
            # WHY: Camera & lighting variation; makes model robust to different lighting conditions
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),

            # Step 6: Random grayscale (10% probability)
            # WHY: Rare but handles b/w video/poor cameras; teaches features beyond color
            transforms.RandomGrayscale(p=0.1),

            # Step 7: Random affine (translation 10%, scale 0.9x to 1.1x)
            # WHY: Small shifts & scale changes from camera motion in video frames
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),

            # Step 8: Convert PIL Image → PyTorch tensor (uint8 [0,255] → float [0,1])
            # DATATYPE: Image shape (H, W, 3) → (3, H, W); pixel values [0,255] → [0.0, 1.0]
            # WHY: PyTorch models expect tensor format, not PIL Images
            transforms.ToTensor(),

            # Step 9: Normalize using ImageNet statistics
            # DATATYPE: float tensor [0, 1] → [-2.1, 2.6] (roughly)
            # WHY: Pretrained ImageNet models expect this normalization; helps numerical stability
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        # VALIDATION & TEST PIPELINE
        # WHY: No augmentation to ensure we evaluate on clean, unmodified images.
        # Data leakage would occur if we randomly rotate test images.
        return transforms.Compose([
            # Step 1: Resize to exact target size
            # DATATYPE: Image → (C, 224, 224)
            # WHY: Model input layer expects fixed size; val/test have no randomness
            transforms.Resize((img_size, img_size)),

            # Step 2: Convert to tensor
            # DATATYPE: PIL Image → torch.Tensor, float [0, 1]
            transforms.ToTensor(),

            # Step 3: Normalize with ImageNet statistics
            # WHY: Same normalization as training for consistency
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class EmotionDataset(Dataset):
    """
    Loads (image, label_id) pairs from a split CSV manifest.

    CSV columns required: path, label, label_id
    """

    def __init__(self, csv_path: str | Path, transform=None):
        # Step 1: Load CSV manifest (produced by dataset_preparation.py)
        # DATATYPE: pd.DataFrame with columns: path (str), label (str), label_id (int)
        # WHY: CSV provides a single source of truth for which images belong to which split
        self.df = pd.read_csv(csv_path)

        # Step 2: Validate that required columns exist
        # WHY: Catch configuration errors early; prevents cryptic KeyError later
        assert 'path'     in self.df.columns, 'CSV missing "path" column'
        assert 'label'    in self.df.columns, 'CSV missing "label" column'
        assert 'label_id' in self.df.columns, 'CSV missing "label_id" column'

        # Step 3: Remove rows where image files no longer exist on disk
        # DATATYPE: valid is bool Series; ~valid inverts the boolean mask
        # WHY: Prevents FileNotFoundError during training if files were deleted/moved
        valid = self.df['path'].apply(lambda p: Path(p).exists())
        n_missing = (~valid).sum()
        if n_missing > 0:
            print(f'[WARN] EmotionDataset: dropping {n_missing} missing files.')
        self.df = self.df[valid].reset_index(drop=True)

        # Step 4: Store transform function for use in __getitem__
        # DATATYPE: transforms.Compose or None
        # WHY: Applied per-sample lazily during iteration (not all at once at load time)
        self.transform = transform

        # Step 5: Extract unique emotion class names and count them
        # DATATYPE: classes is list of strings; num_classes is int
        # WHY: Needed for model output layer size; also for displaying class info
        self.classes   = sorted(self.df['label'].unique())
        self.num_classes = len(self.classes)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def class_weights(self) -> torch.Tensor:
        """
        Returns per-sample weights for WeightedRandomSampler.
        Upsamples minority classes to address class imbalance.
        """
        # Step 1: Count how many samples belong to each class
        # DATATYPE: pd.Series with class_id (int) as index, count as values
        # WHY: If "angry" has 1000 samples but "surprise" has 200, we need to
        # oversample surprise or undersample anger to balance the training distribution
        counts = self.df['label_id'].value_counts().sort_index()

        # Step 2: Convert counts to frequencies (proportion of dataset)
        # DATATYPE: pd.Series, values sum to 1.0
        # WHY: Normalizes counts so we can invert them
        freq   = counts / counts.sum()

        # Step 3: Compute inverse frequencies as class weights
        # DATATYPE: pd.Series with values > 1.0
        # Example: If "anger" is 50% of data, weight = 1/0.5 = 2.0 (common class)
        #          If "surprise" is 5% of data, weight = 1/0.05 = 20.0 (rare class)
        # WHY: Rare classes get higher weight, so the sampler picks them more often
        weight_per_class = 1.0 / freq

        # Step 4: Map each sample to its class weight
        # DATATYPE: numpy array of floats, same length as dataset
        # WHY: WeightedRandomSampler needs one weight per sample, not per class
        sample_weights   = self.df['label_id'].map(weight_per_class).values

        # Step 5: Convert to PyTorch tensor for DataLoader compatibility
        # DATATYPE: torch.Tensor shape (N,) where N = num_samples
        return torch.tensor(sample_weights, dtype=torch.float)

    # ── Standard Dataset interface ────────────────────────────────────────────

    def __len__(self) -> int:
        # WHY: Required by torch.utils.data.Dataset; allows len(dataset)
        return len(self.df)

    def __getitem__(self, idx: int):
        # Step 1: Fetch the CSV row for this sample index
        # DATATYPE: pd.Series containing columns from the CSV for one row
        row   = self.df.iloc[idx]

        # Step 2: Load the image file from disk and ensure RGB format
        # DATATYPE: PIL.Image.Image object
        # WHY: Image.open() preserves original format (PNG, JPEG, etc.);
        #      convert('RGB') ensures 3 channels (handles RGBA, grayscale, etc.)
        image = Image.open(row['path']).convert('RGB')

        # Step 3: Apply augmentation/normalization transforms if provided
        # DATATYPE: torch.Tensor with shape (3, 224, 224) and dtype float32
        # WHY: Transforms convert PIL → tensor and normalize pixel values
        if self.transform:
            image = self.transform(image)

        # Step 4: Extract the emotion class label (0-indexed integer)
        # DATATYPE: int (0 to num_classes-1)
        # WHY: Cross-entropy loss expects integer class indices, not strings
        label = int(row['label_id'])

        # Step 5: Return tuple that DataLoader expects
        # DATATYPE: (torch.Tensor, int)
        return image, label

    def __repr__(self) -> str:
        # WHY: Allows print(dataset) to show informative summary
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
    # Step 1: Sample blending ratio λ from Beta distribution
    # DATATYPE: float in [0, 1]
    # WHY: Beta(α, α) produces values near 0 and 1 (less blending) more often
    #      than near 0.5 (equal blend); matches the paper's recommendation
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    # Step 2: Get batch size to create random permutation
    # DATATYPE: int = x.size(0)
    batch_size = x.size(0)

    # Step 3: Create random permutation to select which samples to mix with
    # DATATYPE: torch.Tensor of ints, same device as x (CPU or GPU)
    # WHY: On GPU, randperm must be on GPU to avoid data transfer overhead
    index = torch.randperm(batch_size, device=x.device)

    # Step 4: Linear interpolation: λ*x + (1-λ)*shuffled_x
    # DATATYPE: torch.Tensor, same shape as x (N, C, H, W)
    # Example: If λ=0.7, sample i becomes 70% original + 30% sample at index[i]
    # WHY: Creates synthetic training data; prevents overfitting to exact images
    mixed_x = lam * x + (1 - lam) * x[index]

    # Step 5: Return mixed images, original labels, shuffled labels, and blend ratio
    # WHY: Loss will blend the two label losses with λ and (1-λ) as weights
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Computes MixUp loss as weighted sum of two CE losses."""
    # Step 1: Compute loss for both label targets weighted by blend ratio
    # DATATYPE: scalar torch.Tensor (single loss value)
    # WHY: If λ=0.7, we're saying the mixed image is 70% of class y_a
    #      so its loss should be 70% of the y_a loss
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x: torch.Tensor, y: torch.Tensor,
                alpha: float = 1.0) -> tuple:
    """
    CutMix augmentation (Yun et al., 2019).
    Cuts and pastes random rectangular regions between images.

    Returns: mixed_x, y_a, y_b, lambda
    """
    # Step 1: Sample blend ratio from Beta distribution
    # DATATYPE: float in [0, 1]
    lam = np.random.beta(alpha, alpha)

    # Step 2: Extract batch dimensions
    # DATATYPE: batch_size is int; H, W are ints (image height, width)
    # Example: x.size() = (32, 3, 224, 224) → batch=32, H=224, W=224
    batch_size, _, H, W = x.size()

    # Step 3: Create random permutation (which samples to mix with)
    # DATATYPE: torch.Tensor of ints on same device as x
    index = torch.randperm(batch_size, device=x.device)

    # Step 4: Calculate cut size from blend ratio
    # DATATYPE: int (pixels)
    # WHY: If λ=0.7, keep 70% of original, cut out √(1-0.7)² area
    cut_rat = np.sqrt(1 - lam)
    cut_w   = int(W * cut_rat)
    cut_h   = int(H * cut_rat)

    # Step 5: Pick random center point for the cut region
    # DATATYPE: int (pixel coordinate)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Step 6: Compute bounding box for the cut, clipped to image bounds
    # DATATYPE: int (pixel coordinates)
    # WHY: np.clip prevents going outside [0, H) and [0, W) ranges
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    # Step 7: Clone the batch and paste the cut region from shuffled batch
    # DATATYPE: torch.Tensor, same shape as x
    # WHY: Clone prevents in-place modification of original data
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Step 8: Recalculate lambda based on actual cut size (handles clipping)
    # DATATYPE: float in [0, 1]
    # Example: If we wanted to cut 30% but clipping reduced it to 20%,
    #          λ is updated to reflect actual blend ratio
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
    # Step 1: Convert to Path object for consistency
    # DATATYPE: Path object
    splits_dir = Path(splits_dir)

    # Step 2: Initialize dictionary to store loaders
    # DATATYPE: dict[str, DataLoader]
    loaders    = {}

    # Step 3: Iterate over each split type (train, val, test)
    for split in ('train', 'val', 'test'):
        # Step 4: Construct CSV path from mode and split name
        # Example: 'ekman6' + 'train' → 'ekman6_train.csv'
        csv_path = splits_dir / f'{mode}_{split}.csv'

        # Step 5: Skip if split file doesn't exist (test is optional)
        # WHY: Not all datasets have a test split; val/test are treated the same way
        if not csv_path.exists():
            continue

        # Step 6: Get appropriate transforms (augmentation for train, none for val/test)
        # DATATYPE: transforms.Compose
        transform = get_transforms(split, img_size)

        # Step 7: Create Dataset object from CSV manifest
        # DATATYPE: EmotionDataset
        # WHY: Dataset handles image loading, label mapping, and transforms per-sample
        dataset   = EmotionDataset(csv_path, transform=transform)

        # Step 8: For training split, use WeightedRandomSampler to handle imbalance
        if split == 'train' and use_sampler:
            # WHY: If dataset has 1000 "happy" and 100 "surprise", sampler
            #      ensures batches include surprise proportionally more often
            # DATATYPE: WeightedRandomSampler
            sampler = WeightedRandomSampler(
                weights     = dataset.class_weights,  # Higher weight = sampled more
                num_samples = len(dataset),           # Always sample full dataset
                replacement = True                    # Allow repeats (required for sampling)
            )

            # Step 9: Create DataLoader with custom sampler (overrides shuffle=True)
            # DATATYPE: DataLoader
            # WHY: sampler parameter prevents shuffle; pin_memory speeds up GPU transfer
            loader = DataLoader(
                dataset,
                batch_size  = batch_size,
                sampler     = sampler,              # Use weighted sampling instead of random
                num_workers = num_workers,          # Parallel CPU workers for image loading
                pin_memory  = True,                 # Pre-allocate GPU memory for tensors
            )
        else:
            # Step 10: For val/test, use default sequential sampler
            # WHY: No need to balance or augment; evaluate on all data in order
            loader = DataLoader(
                dataset,
                batch_size  = batch_size,
                shuffle     = False,                # Don't shuffle val/test
                num_workers = num_workers,
                pin_memory  = True,
            )

        # Step 11: Store loader in dictionary
        loaders[split] = loader

        # Step 12: Print summary of dataset size and classes
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
    # Step 1: Load all splits with small batch size for quick test
    # WHY: batch_size=8 and num_workers=0 keep memory low during testing
    loaders    = get_loaders(splits_dir, mode, batch_size=8, num_workers=0)

    # Step 2: Verify each loader works by fetching one batch
    for name, loader in loaders.items():
        # DATATYPE: imgs is torch.Tensor (8, 3, 224, 224); labels is torch.Tensor (8,)
        # WHY: If this runs without error, transforms and CSV are valid
        imgs, labels = next(iter(loader))
        print(f'{name}: images={imgs.shape}, labels={labels}')
