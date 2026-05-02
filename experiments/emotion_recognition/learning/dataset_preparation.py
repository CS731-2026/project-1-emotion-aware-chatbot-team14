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
# Step 1: Set random seeds for all libraries
# WHY: Ensures the same train/val split is produced every time.
#      Critical for reproducible research and comparing models fairly.
random.seed(42)
np.random.seed(42)

# ── Label definitions ─────────────────────────────────────────────────────────
# Step 2: Define which emotion classes to use in different modes
# DATATYPE: list of strings
# WHY: Ekman's basic 6 emotions are the most well-established in psychology.
#      ekman7 adds neutral (important for real-world video).
#      all8 includes contempt (rare, hard to classify).
ALL_8_EMOTIONS = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EKMAN_6        = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise']   # no neutral/contempt
EKMAN_7        = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # no contempt


# ── 1. Load labels.csv ────────────────────────────────────────────────────────
def load_labels_csv(dataset_dir: Path) -> pd.DataFrame:
    """
    Load the labels.csv produced by AffectNetHQ preprocessing.
    Expected columns: pth, label, relFCs  (relFCs = confidence score)
    """
    # Step 1: Construct path to labels.csv
    # DATATYPE: Path object
    csv_path = dataset_dir / 'labels.csv'

    # Step 2: Exit immediately if labels.csv doesn't exist
    # WHY: Fail fast rather than with cryptic errors later during iteration
    if not csv_path.exists():
        sys.exit(f'[ERROR] labels.csv not found at {csv_path}')

    # Step 3: Load CSV with index (first column becomes row labels)
    # DATATYPE: pd.DataFrame
    df = pd.read_csv(csv_path, index_col=0)

    # Step 4: Normalize column names to lowercase and remove whitespace
    # WHY: Different data sources may have 'Pth', 'PATH', 'pth ', etc.
    #      Normalization prevents KeyError when accessing columns
    df.columns = [c.strip().lower() for c in df.columns]

    # Step 5: Rename non-standard column names to expected names
    # WHY: 'pth' is AffectNetHQ's name; we use 'path' for consistency
    if 'pth' in df.columns:
        df.rename(columns={'pth': 'path'}, inplace=True)

    # WHY: 'relFCs' = relative face confidence; rename for clarity
    if 'relfcs' in df.columns:
        df.rename(columns={'relfcs': 'confidence'}, inplace=True)

    # Step 6: Convert relative paths to absolute paths
    # DATATYPE: str (file paths)
    # WHY: CSV contains relative paths like "anger/image_001.jpg"
    #      We need absolute paths to load files from anywhere
    df['path'] = df['path'].str.replace('\\', '/', regex=False)  # Handle Windows paths
    df['path'] = df['path'].apply(lambda p: (dataset_dir / p).resolve().as_posix())

    # Step 7: Standardize emotion labels to lowercase
    # WHY: 'Happy', 'HAPPY', 'happy' should all be treated identically
    df['label'] = df['label'].str.strip().str.lower()

    print(f'[INFO] Loaded {len(df):,} rows from labels.csv')
    return df


# ── 2. Validate files ─────────────────────────────────────────────────────────
def remove_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose image file does not exist on disk."""
    # Step 1: Find rows where file doesn't exist
    # DATATYPE: pd.Series of bools (True where file missing)
    missing = df[~df['path'].apply(lambda p: Path(p).exists())]

    # Step 2: Warn and drop missing rows
    # WHY: Files may have been deleted/moved between dataset creation and training.
    #      Training on missing files would crash mid-epoch.
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
    # Step 1: Map mode string to class list
    # DATATYPE: dict mapping str → list of str
    # WHY: Allows clean mapping without long if/elif chains
    keep_map = {'ekman6': EKMAN_6, 'ekman7': EKMAN_7, 'all8': ALL_8_EMOTIONS}
    keep = keep_map.get(mode, EKMAN_6)

    # Step 2: Record original dataset size for reporting
    before = len(df)

    # Step 3: Keep only rows where label is in the selected class list
    # WHY: If mode='ekman6', discard all 'neutral' and 'contempt' samples.
    #      Reduces noise from excluded classes.
    df = df[df['label'].isin(keep)].copy().reset_index(drop=True)

    # Step 4: Print summary of filtering
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
    # Step 1: Check if confidence column exists
    # WHY: Not all datasets provide confidence scores
    if 'confidence' not in df.columns:
        print('[INFO] No confidence column — skipping threshold filter.')
        return df

    # Step 2: Record original size
    before = len(df)

    # Step 3: Keep only high-confidence samples
    # DATATYPE: confidence values are floats in [0, 1]
    # WHY: Low-confidence detections are likely misannotated or ambiguous faces.
    #      But in practice, using all data (threshold=0.0) gave best results
    #      because dataset size matters more than annotation quality.
    df = df[df['confidence'] >= threshold].reset_index(drop=True)

    print(f'[INFO] Confidence ≥ {threshold}: kept {len(df):,} / {before:,} images')
    return df


# ── 5. Print class distribution ───────────────────────────────────────────────
def print_distribution(df: pd.DataFrame, title: str = 'Distribution'):
    """Print a visual histogram of class counts."""
    print(f'\n── {title} ──')

    # Step 1: Count samples per class
    # DATATYPE: pd.Series with emotion names as index, counts as values
    counts = df['label'].value_counts().sort_index()

    # Step 2: Print bar chart (one bar per emotion)
    # WHY: Visual representation helps spot severe imbalance at a glance
    for label, n in counts.items():
        # Each █ represents 100 samples; helps visualize proportions
        bar = '█' * (n // 100)
        print(f'  {label:10s}: {n:>5}  {bar}')

    # Step 3: Print total
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
    # Step 1: Extract labels for stratification
    # DATATYPE: pd.Series of emotion labels (str)
    # WHY: stratify parameter ensures each split has the same class proportions as the full dataset
    labels = df['label']

    # Step 2: Check if 3-way split (train/val/test) is requested
    if test_size > 0:
        # STEP 2A: 3-way split using two consecutive train_test_split calls
        # WHY: sklearn doesn't have native 3-way split; we use nested 2-way splits

        # First split: separate test+val from train
        # DATATYPE: Two pd.DataFrames
        train_df, temp_df = train_test_split(
            df,
            test_size=val_size + test_size,  # e.g., 0.30 total for val+test
            stratify=labels,                 # Preserve class distribution
            random_state=random_state        # Reproducible split
        )

        # Step 2B: Calculate what fraction of temp_df should be val vs test
        # Example: if we want 15% val and 15% test, and temp_df is 30% of total,
        #          then within temp_df, val should be 50% and test should be 50%
        val_frac = val_size / (val_size + test_size)

        # Step 2C: Split the temp group into val and test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_frac,      # e.g., 0.5 of temp becomes test
            stratify=temp_df['label'],   # Preserve class distribution
            random_state=random_state
        )

        # Step 2D: Build output dictionary with all three splits
        splits = {'train': train_df.reset_index(drop=True),
                  'val':   val_df.reset_index(drop=True),
                  'test':  test_df.reset_index(drop=True)}
    else:
        # STEP 3: 2-way split (train/val only)
        # WHY: Not all datasets have a reserved test set
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,      # e.g., 0.15
            stratify=labels,
            random_state=random_state
        )

        splits = {'train': train_df.reset_index(drop=True),
                  'val':   val_df.reset_index(drop=True)}

    # Step 4: Print split sizes and class distributions within each split
    # WHY: Verify that stratification worked (each split has similar class proportions)
    for name, sdf in splits.items():
        print(f'  {name:6s}: {len(sdf):>6,} images  '
              f'({sdf["label"].value_counts().sort_index().to_dict()})')

    return splits


# ── 7. Save split CSVs ────────────────────────────────────────────────────────
def save_splits(splits: dict, output_dir: Path, mode: str):
    """Save each split as a CSV manifest ready for the PyTorch Dataset class."""
    # Step 1: Create output directory if it doesn't exist
    # WHY: Ensures the directory structure is ready before writing files
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Save each split as a CSV file
    # DATATYPE: CSV files with columns: path, label, label_id
    # WHY: These CSVs will be loaded by EmotionDataset during training.
    #      CSV format is universal and human-readable for debugging.
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
    # Step 1: Try to import video/face-detection libraries
    # WHY: These are optional; many users won't need EmoCare extraction
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError:
        print('[ERROR] cv2 and ultralytics required for EmoCare extraction.')
        return

    # Step 2: Load YOLOv8 nano face detection model
    # DATATYPE: YOLO model object
    # WHY: Detects faces in video frames so we can crop tight regions
    face_model = YOLO('yolov8n-face.pt')

    # Step 3: Iterate over emotion directories (anger/, happy/, etc.)
    # DATATYPE: Path objects
    for emotion_dir in sorted(emoCare_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue

        # Step 4: Extract emotion name and create output directory
        emotion = emotion_dir.name.lower()
        out_emotion_dir = output_dir / emotion
        out_emotion_dir.mkdir(parents=True, exist_ok=True)

        # Step 5: Find all video files in this emotion directory
        # DATATYPE: list of Path objects (mp4 and avi files)
        video_files = list(emotion_dir.glob('*.mp4')) + list(emotion_dir.glob('*.avi'))
        print(f'[EmoCare] {emotion}: {len(video_files)} videos')

        # Step 6: Process each video file
        for i, video_path in enumerate(video_files):
            # Step 6A: Open video file
            # DATATYPE: cv2.VideoCapture object
            # WHY: cv2 is OpenCV; standard library for video processing
            cap = cv2.VideoCapture(str(video_path))

            # Step 6B: Get total frame count
            # DATATYPE: int (number of frames)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Step 6C: Skip videos with no frames (corrupted/empty)
            if total == 0:
                cap.release()
                continue

            # Step 6D: Seek to middle frame
            # WHY: Middle frame is most representative of the emotion
            #      (avoids fade-in/out at video start/end)
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)

            # Step 6E: Read the frame
            # DATATYPE: bool (success), np.ndarray (BGR image, uint8)
            # WHY: Check 'ret' to ensure frame was actually read
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                continue

            # Step 6F: Detect faces in the frame using YOLOv8
            # DATATYPE: YOLO Results object with bounding boxes
            results = face_model(frame, verbose=False)[0]

            # Step 6G: Extract best (most confident) face detection
            # WHY: If multiple faces detected, use the one with highest confidence
            if results.boxes is not None and len(results.boxes) > 0:
                best = results.boxes.conf.argmax().item()  # Index of highest confidence
                x1, y1, x2, y2 = results.boxes.xyxy[best].int().tolist()  # Bounding box coords

                # Step 6H: Add padding around face for context
                # WHY: Face-only crops can be too tight; padding provides context
                h, w = frame.shape[:2]  # Frame height, width
                pad = 10
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)  # Expand box, clip to image bounds
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

                # Step 6I: Crop face region
                # DATATYPE: np.ndarray (subset of frame)
                face = frame[y1:y2, x1:x2]
            else:
                # Step 6J: If no face detected, use full frame
                # WHY: Better to have full frame than no image
                face = frame

            # Step 6K: Save the face crop to disk
            # DATATYPE: JPEG image file
            # WHY: JPEG compression keeps file sizes reasonable
            out_path = out_emotion_dir / f'emocare_{i:04d}.jpg'
            cv2.imwrite(str(out_path), face)

    print(f'[EmoCare] Extraction complete → {output_dir}')


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='CS731 Dataset Preparation')

    # WHY: Dataset root directory must be provided by user
    parser.add_argument('--dataset_dir',  type=str,   default='1_Dataset',
                        help='Path to the dataset root (contains labels.csv)')

    # WHY: User controls where processed manifests are saved
    parser.add_argument('--output_dir',   type=str,   default='data/splits',
                        help='Where to save train/val/test CSVs')

    # WHY: Different mode selections affect model architecture and comparisons
    parser.add_argument('--mode',         type=str,   default='ekman6',
                        choices=['ekman6', 'ekman7', 'all8'],
                        help='Which emotion classes to keep')

    # WHY: Confidence threshold optional; exemplars found all-data gives best results
    parser.add_argument('--conf_threshold', type=float, default=0.0,
                        help='Min AffectNetHQ confidence (0.0 = use all data; '
                             'exemplars found all-data gives best accuracy)')

    # WHY: Validation split size affects train/val tradeoff
    parser.add_argument('--val_size',     type=float, default=0.15,
                        help='Fraction of data for validation (default: 0.15)')

    # WHY: Test set is optional; some projects don't have reserved test data
    parser.add_argument('--test_size',    type=float, default=0.0,
                        help='Fraction of data for test set (0 = no test set)')

    return parser.parse_args()


def main():
    # Step 1: Parse command-line arguments
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)

    # Step 2: Print banner
    print('\n' + '='*60)
    print('  CS731 Dataset Preparation')
    print('='*60)

    # STEP 3: Load raw labels.csv
    # WHY: All subsequent filtering depends on this
    df = load_labels_csv(dataset_dir)

    # STEP 4: Validate that all referenced files exist
    # WHY: Catch missing files early rather than during training
    df = remove_missing(df)

    # STEP 5: Filter to desired emotion classes
    # WHY: Ekman-6 is the most standard setup; others for comparison
    df, kept_classes = filter_classes(df, args.mode)

    # STEP 6: Apply confidence threshold if requested
    # WHY: Low-confidence annotations may be mislabeled
    if args.conf_threshold > 0.0:
        df = apply_confidence_threshold(df, args.conf_threshold)

    # STEP 7: Show final class distribution before splitting
    # WHY: Verify that classes are reasonably balanced
    print_distribution(df, title=f'Final distribution ({args.mode})')

    # STEP 8: Create integer label IDs (0, 1, 2, ..., num_classes-1)
    # DATATYPE: dict mapping emotion string → int; added as new column to df
    # WHY: PyTorch models need integer class indices, not strings
    label_to_id = {label: i for i, label in enumerate(sorted(kept_classes))}
    df['label_id'] = df['label'].map(label_to_id)
    print(f'[INFO] Label mapping: {label_to_id}')

    # STEP 9: Stratified split (preserve class distribution in each split)
    # WHY: If train is 80% happy and 20% angry, val should be too
    #      (not 50% happy by random chance)
    print(f'\n[INFO] Splitting: val={args.val_size}, test={args.test_size}')
    splits = stratified_split(df, args.val_size, args.test_size)

    # STEP 10: Save split manifests as CSVs
    # WHY: EmotionDataset loads images using these CSVs during training
    save_splits(splits, output_dir, args.mode)

    # STEP 11: Print final summary
    print('\n✅ Dataset preparation complete.')
    print(f'   Manifests saved to: {output_dir.resolve()}')
    print('   Next step: run  python models/train.py --mode', args.mode)


if __name__ == '__main__':
    main()
