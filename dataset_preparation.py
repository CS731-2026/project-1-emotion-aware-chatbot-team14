"""
CS731 — Dataset Preparation Script
===================================
Handles:
  1. Reading and validating labels.csv from 1_Dataset/
  2. Filtering classes (keep Ekman's 6 or 7; drop contempt / neutral)
  3. Optionally applying AffectNetHQ confidence thresholds
  4. Extracting middle frames from EmoCare videos
  5. Stratified train/val(/test) split
  6. Saving split manifests as CSV files

Usage
-----
  python data/dataset_preparation.py --dataset_dir 1_Dataset --mode ekman6
  python data/dataset_preparation.py --dataset_dir 1_Dataset --mode ekman7
  python data/dataset_preparation.py --dataset_dir 1_Dataset --mode all8

Directory expected
------------------
  1_Dataset/
  ├── labels.csv          (columns: pth, label, relFCs)
  ├── anger/
  ├── contempt/
  ├── disgust/
  ├── fear/
  ├── happy/
  ├── neutral/
  ├── sad/
  └── surprise/
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Label definitions ─────────────────────────────────────────────────────────
ALL_8_EMOTIONS = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EKMAN_6        = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise']   # no neutral/contempt
EKMAN_7        = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # no contempt


# ── 1. Load labels.csv ────────────────────────────────────────────────────────
def load_labels_csv(dataset_dir: Path) -> pd.DataFrame:
    """
    Load the labels.csv produced by AffectNetHQ preprocessing.
    Expected columns: pth, label, relFCs  (relFCs = confidence score)
    """
    csv_path = dataset_dir / 'labels.csv'
    if not csv_path.exists():
        sys.exit(f'[ERROR] labels.csv not found at {csv_path}')

    df = pd.read_csv(csv_path, index_col=0)

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    if 'pth' in df.columns:
        df.rename(columns={'pth': 'path'}, inplace=True)
    if 'relfcs' in df.columns:
        df.rename(columns={'relfcs': 'confidence'}, inplace=True)

    # Make paths absolute (handle both / and \ separators)
    df['path'] = df['path'].str.replace('\\', '/', regex=False)
    df['path'] = df['path'].apply(lambda p: (dataset_dir / p).resolve().as_posix())

    # Standardise label names to lowercase
    df['label'] = df['label'].str.strip().str.lower()

    print(f'[INFO] Loaded {len(df):,} rows from labels.csv')
    return df


# ── 2. Validate files ─────────────────────────────────────────────────────────
def remove_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose image file does not exist on disk."""
    missing = df[~df['path'].apply(lambda p: Path(p).exists())]
    if len(missing) > 0:
        print(f'[WARN] Dropping {len(missing)} rows with missing files.')
        df = df[df['path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    else:
        print('[INFO] All image files verified ✓')
    return df


# ── 3. Filter by emotion mode ─────────────────────────────────────────────────
def filter_classes(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode: 'ekman6' | 'ekman7' | 'all8'
    """
    keep_map = {'ekman6': EKMAN_6, 'ekman7': EKMAN_7, 'all8': ALL_8_EMOTIONS}
    keep = keep_map.get(mode, EKMAN_6)
    before = len(df)
    df = df[df['label'].isin(keep)].copy().reset_index(drop=True)
    print(f'[INFO] Mode={mode}: kept {len(df):,} / {before:,} images '
          f'({", ".join(keep)})')
    return df, keep


# ── 4. Optional: AffectNetHQ confidence threshold ────────────────────────────
def apply_confidence_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    If the CSV has a 'confidence' column, drop rows below `threshold`.
    Both exemplars found that removing the threshold (using all data) gave
    better accuracy due to larger dataset size.
    """
    if 'confidence' not in df.columns:
        print('[INFO] No confidence column — skipping threshold filter.')
        return df
    before = len(df)
    df = df[df['confidence'] >= threshold].reset_index(drop=True)
    print(f'[INFO] Confidence ≥ {threshold}: kept {len(df):,} / {before:,} images')
    return df


# ── 5. Print class distribution ───────────────────────────────────────────────
def print_distribution(df: pd.DataFrame, title: str = 'Distribution'):
    print(f'\n── {title} ──')
    counts = df['label'].value_counts().sort_index()
    for label, n in counts.items():
        bar = '█' * (n // 100)
        print(f'  {label:10s}: {n:>5}  {bar}')
    print(f'  {"TOTAL":10s}: {len(df):>5}')
    print()


# ── 6. Stratified split ───────────────────────────────────────────────────────
def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.0,   # set to 0.15 for a 3-way split
    random_state: int = 42
) -> dict:
    """
    Returns {'train': df, 'val': df} or {'train': df, 'val': df, 'test': df}.
    Uses stratified sampling to preserve class balance in all splits.

    Group 15 (Frienderly):  85/15  (val_size=0.15, test_size=0.0)
    Team 7   (ChatBox):     70/15/15 (val_size=0.15, test_size=0.15)
    """
    labels = df['label']

    if test_size > 0:
        # 3-way split: train / val / test
        train_df, temp_df = train_test_split(
            df, test_size=val_size + test_size,
            stratify=labels, random_state=random_state
        )
        val_frac = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df, test_size=1 - val_frac,
            stratify=temp_df['label'], random_state=random_state
        )
        splits = {'train': train_df.reset_index(drop=True),
                  'val':   val_df.reset_index(drop=True),
                  'test':  test_df.reset_index(drop=True)}
    else:
        # 2-way split: train / val
        train_df, val_df = train_test_split(
            df, test_size=val_size,
            stratify=labels, random_state=random_state
        )
        splits = {'train': train_df.reset_index(drop=True),
                  'val':   val_df.reset_index(drop=True)}

    for name, sdf in splits.items():
        print(f'  {name:6s}: {len(sdf):>6,} images  '
              f'({sdf["label"].value_counts().sort_index().to_dict()})')

    return splits


# ── 7. Save split CSVs ────────────────────────────────────────────────────────
def save_splits(splits: dict, output_dir: Path, mode: str):
    """Save each split as a CSV manifest ready for the PyTorch Dataset class."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, df in splits.items():
        out_path = output_dir / f'{mode}_{split_name}.csv'
        df.to_csv(out_path, index=False)
        print(f'[INFO] Saved {split_name} manifest → {out_path}')


# ── 8. (Optional) EmoCare video → middle frame extraction ────────────────────
def extract_emoCare_frames(emoCare_dir: Path, output_dir: Path):
    """
    EmoCare dataset is a collection of labelled video clips.
    This extracts the middle frame of each video, runs YOLOv8n-face to
    crop to the face, and saves to output_dir/<emotion>/frame_<n>.jpg

    Usage: called separately, not part of the main pipeline.
    """
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError:
        print('[ERROR] cv2 and ultralytics required for EmoCare extraction.')
        return

    face_model = YOLO('yolov8n-face.pt')

    for emotion_dir in sorted(emoCare_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue
        emotion = emotion_dir.name.lower()
        out_emotion_dir = output_dir / emotion
        out_emotion_dir.mkdir(parents=True, exist_ok=True)

        video_files = list(emotion_dir.glob('*.mp4')) + list(emotion_dir.glob('*.avi'))
        print(f'[EmoCare] {emotion}: {len(video_files)} videos')

        for i, video_path in enumerate(video_files):
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total == 0:
                cap.release()
                continue

            # Jump to middle frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                continue

            # Detect and crop face
            results = face_model(frame, verbose=False)[0]
            if results.boxes is not None and len(results.boxes) > 0:
                best = results.boxes.conf.argmax().item()
                x1, y1, x2, y2 = results.boxes.xyxy[best].int().tolist()
                h, w = frame.shape[:2]
                pad = 10
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                face = frame[y1:y2, x1:x2]
            else:
                face = frame  # no face found — save full frame

            out_path = out_emotion_dir / f'emocare_{i:04d}.jpg'
            cv2.imwrite(str(out_path), face)

    print(f'[EmoCare] Extraction complete → {output_dir}')


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='CS731 Dataset Preparation')
    parser.add_argument('--dataset_dir',  type=str,   default='1_Dataset',
                        help='Path to the dataset root (contains labels.csv)')
    parser.add_argument('--output_dir',   type=str,   default='data/splits',
                        help='Where to save train/val/test CSVs')
    parser.add_argument('--mode',         type=str,   default='ekman6',
                        choices=['ekman6', 'ekman7', 'all8'],
                        help='Which emotion classes to keep')
    parser.add_argument('--conf_threshold', type=float, default=0.0,
                        help='Min AffectNetHQ confidence (0.0 = use all data; '
                             'exemplars found all-data gives best accuracy)')
    parser.add_argument('--val_size',     type=float, default=0.15,
                        help='Fraction of data for validation (default: 0.15)')
    parser.add_argument('--test_size',    type=float, default=0.0,
                        help='Fraction of data for test set (0 = no test set)')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)

    print('\n' + '='*60)
    print('  CS731 Dataset Preparation')
    print('='*60)

    # Step 1: Load
    df = load_labels_csv(dataset_dir)

    # Step 2: Validate
    df = remove_missing(df)

    # Step 3: Filter classes
    df, kept_classes = filter_classes(df, args.mode)

    # Step 4: Confidence threshold
    if args.conf_threshold > 0.0:
        df = apply_confidence_threshold(df, args.conf_threshold)

    # Step 5: Inspect distribution
    print_distribution(df, title=f'Final distribution ({args.mode})')

    # Step 6: Create label_id column (integer class index)
    label_to_id = {label: i for i, label in enumerate(sorted(kept_classes))}
    df['label_id'] = df['label'].map(label_to_id)
    print(f'[INFO] Label mapping: {label_to_id}')

    # Step 7: Split
    print(f'\n[INFO] Splitting: val={args.val_size}, test={args.test_size}')
    splits = stratified_split(df, args.val_size, args.test_size)

    # Step 8: Save
    save_splits(splits, output_dir, args.mode)

    print('\n✅ Dataset preparation complete.')
    print(f'   Manifests saved to: {output_dir.resolve()}')
    print('   Next step: run  python models/train.py --mode', args.mode)


if __name__ == '__main__':
    main()
