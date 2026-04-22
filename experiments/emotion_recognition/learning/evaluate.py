"""
CS731 — Model Evaluation Script
=================================
Loads a saved checkpoint and evaluates on the val (or test) split.
Produces:
  - Overall accuracy & loss
  - Per-class precision, recall, F1-score
  - Confusion matrix (heatmap matching exemplar report figures)
  - Inference speed (FPS)

Usage
-----
  python models/evaluate.py --checkpoint models/checkpoints/swin_tiny_ekman6_best.pt
  python models/evaluate.py --checkpoint models/checkpoints/swin_tiny_ekman6_best.pt --split test
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'application' / 'mock_programs'))
from emotion_dataset import EmotionDataset, get_transforms
from model import build_model
from torch.utils.data import DataLoader


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str | Path, device: torch.device):
    """Load a training checkpoint and reconstruct the model."""
    # Step 1: Load checkpoint file from disk
    # DATATYPE: dict with keys: 'model_name', 'mode', 'num_classes', 'state_dict', etc.
    # WHY: map_location ensures tensors are loaded to the correct device (GPU/CPU)
    ckpt = torch.load(ckpt_path, map_location=device)

    # Step 2: Extract metadata from checkpoint
    # DATATYPE: strings and ints
    # WHY: Need these to recreate the exact model architecture
    model_name  = ckpt['model_name']
    num_classes = ckpt['num_classes']
    epoch       = ckpt['epoch']
    val_acc     = ckpt.get('val_acc', '?')

    # Step 3: Print checkpoint info
    print(f'[INFO] Loading checkpoint: {ckpt_path}')
    print(f'       Model: {model_name}  |  Classes: {num_classes}  |  '
          f'Epoch: {epoch}  |  Val Acc: {val_acc:.2f}%')

    # Step 4: Rebuild model architecture
    # DATATYPE: nn.Module
    # WHY: pretrained=False because we'll load weights from checkpoint
    #      (don't want to waste time loading ImageNet weights we'll overwrite)
    model = build_model(model_name, num_classes=num_classes, pretrained=False)

    # Step 5: Load saved weights into the model
    # WHY: state_dict contains all the learned parameters from training
    model.load_state_dict(ckpt['state_dict'])

    # Step 6: Move model to device and set to eval mode
    # WHY: eval() disables dropout and uses fixed batch norm statistics
    model = model.to(device)
    model.eval()

    return model, ckpt


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()  # Disable gradient computation (evaluation doesn't need backprop)
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device, criterion: nn.Module) -> dict:
    """
    Full evaluation pass. Returns dict with all metrics.
    """
    # Step 1: Initialize lists to accumulate predictions and labels
    # DATATYPE: list of ints
    # WHY: We'll concatenate all batches' predictions to compute aggregate metrics
    all_preds  = []
    all_labels = []

    # Step 2: Initialize accumulators for loss and timing
    # DATATYPE: floats
    total_loss = 0.0
    total_time = 0.0
    n_batches  = 0

    # Step 3: Iterate over evaluation batches
    for images, labels in loader:
        # Step 4: Move data to device (GPU/CPU)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Step 5: Record inference start time
        # WHY: Computing FPS (frames per second) for performance benchmarking
        t0      = time.perf_counter()

        # Step 6: Forward pass (model prediction without gradient computation)
        # DATATYPE: outputs is torch.Tensor (batch_size, num_classes) with logits
        outputs = model(images)

        # Step 7: Record inference end time and accumulate
        # DATATYPE: float (seconds)
        total_time += (time.perf_counter() - t0)

        # Step 8: Compute loss on this batch
        # DATATYPE: scalar tensor
        # WHY: Loss tells us how confident/wrong the predictions were
        loss        = criterion(outputs, labels)
        total_loss += loss.item()
        n_batches  += 1

        # Step 9: Get predicted class (argmax of logits)
        # DATATYPE: torch.Tensor of ints (batch_size,)
        # WHY: outputs.max(1) returns (values, indices); we only need indices
        _, preds = outputs.max(1)

        # Step 10: Convert predictions and labels to numpy and accumulate
        # DATATYPE: numpy arrays
        # WHY: sklearn metrics expect numpy arrays, not PyTorch tensors
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Step 11: Convert accumulated lists to numpy arrays
    # DATATYPE: np.ndarray of ints
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Step 12: Compute average loss over all batches
    # DATATYPE: float
    avg_loss = total_loss / n_batches

    # Step 13: Compute overall accuracy (correct predictions / total)
    # DATATYPE: float (percentage)
    accuracy = 100.0 * (all_preds == all_labels).mean()

    # Step 14: Compute inference speed (samples per second)
    # DATATYPE: float (samples/second)
    # WHY: FPS = total samples / total inference time
    fps      = len(all_labels) / total_time

    # Step 15: Return comprehensive evaluation results
    # DATATYPE: dict with keys for all metrics
    return {
        'loss':       avg_loss,
        'accuracy':   accuracy,
        'fps':        fps,
        'preds':      all_preds,
        'labels':     all_labels,
    }


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray,
                           class_names: list, save_path: str | Path,
                           model_name: str = '') -> None:
    """
    Plots a normalised confusion matrix with raw counts and percentages,
    matching the style used in both exemplar reports.
    """
    # Step 1: Compute raw confusion matrix
    # DATATYPE: np.ndarray of ints, shape (num_classes, num_classes)
    # WHY: cm[i,j] = number of times class i was predicted as class j
    cm      = confusion_matrix(labels, preds)

    # Step 2: Normalize confusion matrix to percentages per row
    # DATATYPE: np.ndarray of floats
    # WHY: Percentages show what % of each true class was predicted correctly/incorrectly
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Step 3: Create figure with size proportional to number of classes
    # DATATYPE: matplotlib Figure and Axes
    # WHY: Large matrices need bigger figures to be readable
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names))))

    # Step 4: Build annotation strings (count + percentage per cell)
    # DATATYPE: np.ndarray of strings
    # WHY: Shows both raw count and percentage for interpretability
    annot = np.empty_like(cm, dtype=object)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            # Example: "45\n(78.5%)" means 45 samples, which is 78.5% of the row
            annot[i, j] = f'{cm[i,j]}\n{cm_norm[i,j]:.1f}%'

    # Step 5: Create heatmap visualization
    # DATATYPE: matplotlib Axes object
    # WHY: Heatmap uses color intensity to show performance per class
    #      Diagonal (correct predictions) should be dark; off-diagonal (errors) light
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=100, ax=ax, linewidths=0.5, linecolor='gray')

    # Step 6: Label axes
    # WHY: Clarifies which dimension is predicted vs actual
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)

    # Step 7: Create informative title
    # DATATYPE: string
    # WHY: Identifies which model and split the matrix is from
    title = f'Confusion Matrix on {"Test" if "test" in str(save_path) else "Validation"} Set'
    if model_name:
        title = f'{model_name} — {title}'
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Step 8: Rotate axis labels for readability
    # WHY: Diagonal class names are hard to read; rotation makes them horizontal
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)

    # Step 9: Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'[INFO] Confusion matrix saved → {save_path}')


# ── Per-class metrics table ───────────────────────────────────────────────────

def print_classification_report(labels: np.ndarray, preds: np.ndarray,
                                  class_names: list) -> None:
    """Print sklearn classification report (precision, recall, F1 per class)."""
    # Step 1: Generate classification report from sklearn
    # DATATYPE: string (formatted table)
    # WHY: Provides per-class precision, recall, and F1-score for detailed analysis
    report = classification_report(
        labels, preds,
        target_names=class_names,
        digits=3,                  # 3 decimal places for precision
        zero_division=0             # Handle classes with no test samples gracefully
    )

    # Step 2: Print report
    print('\n── Per-Class Metrics ──────────────────────────────────')
    print(report)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description='CS731 Model Evaluation')

    # WHY: User must specify which checkpoint to evaluate
    p.add_argument('--checkpoint',  type=str, required=True,
                   help='Path to .pt checkpoint file')

    # WHY: Need to find the correct split CSV
    p.add_argument('--splits_dir',  type=str, default='data/splits',
                   help='Directory with split CSVs')

    # WHY: Can evaluate on different splits (train shows overfitting, val/test show generalization)
    p.add_argument('--split',       type=str, default='val',
                   choices=['train', 'val', 'test'])

    p.add_argument('--batch_size',  type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--results_dir', type=str, default='results/evaluation')

    return p.parse_args()


def main():
    # Step 1: Parse command-line arguments
    args   = parse_args()

    # Step 2: Detect compute device (GPU/CPU)
    # DATATYPE: torch.device
    # WHY: Evaluation can run on GPU or CPU (GPU is faster)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 3: Create results directory if it doesn't exist
    # WHY: Will save confusion matrix and summary CSV here
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Step 4: Load model from checkpoint
    # DATATYPE: nn.Module and dict
    # WHY: Need to restore exact architecture and weights
    model, ckpt = load_checkpoint(args.checkpoint, device)
    model_name  = ckpt['model_name']
    mode        = ckpt['mode']

    # Step 5: Construct path to split CSV
    # DATATYPE: Path object
    # WHY: Must load data from the same split that was used during training
    csv_path = Path(args.splits_dir) / f'{mode}_{args.split}.csv'

    # Step 6: Check that split CSV exists
    # WHY: If it doesn't, user needs to run dataset_preparation.py first
    if not csv_path.exists():
        raise FileNotFoundError(
            f'Split CSV not found: {csv_path}\n'
            f'Run: python data/dataset_preparation.py --mode {mode}'
        )

    # Step 7: Load dataset from split CSV
    # DATATYPE: EmotionDataset
    # WHY: Always use 'val' transforms for evaluation (no augmentation)
    #      even if evaluating on test split (ensures fair comparison)
    transform = get_transforms('val')
    dataset   = EmotionDataset(csv_path, transform=transform)

    # Step 8: Create DataLoader for efficient batch processing
    # DATATYPE: DataLoader
    loader    = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers,
                           pin_memory=True)

    # Step 9: Extract class names for reporting
    # DATATYPE: list of strings
    class_names = dataset.classes

    # Step 10: Print evaluation setup
    print(f'\n[INFO] Evaluating on {args.split} split '
          f'({len(dataset)} images, {len(class_names)} classes)')

    # Step 11: Run full evaluation
    # DATATYPE: dict with loss, accuracy, fps, predictions, labels
    criterion = nn.CrossEntropyLoss()
    results   = evaluate(model, loader, device, criterion)

    # Step 12: Print overall performance metrics
    print(f'\n── Results: {model_name} ({args.split}) ──────────────────')
    print(f'  Accuracy : {results["accuracy"]:.2f}%')
    print(f'  Loss     : {results["loss"]:.4f}')
    print(f'  Speed    : {results["fps"]:.1f} FPS')

    # Step 13: Print per-class metrics (precision, recall, F1)
    # WHY: Shows which classes the model struggles with
    print_classification_report(results['labels'], results['preds'], class_names)

    # Step 14: Compute macro-averaged metrics
    # DATATYPE: floats
    # WHY: Macro average treats each class equally (good for imbalanced data)
    #      unlike accuracy which weights by class frequency
    mac_p  = precision_score(results['labels'], results['preds'], average='macro', zero_division=0)
    mac_r  = recall_score   (results['labels'], results['preds'], average='macro', zero_division=0)
    mac_f1 = f1_score       (results['labels'], results['preds'], average='macro', zero_division=0)
    print(f'Macro Precision: {mac_p:.3f}  Recall: {mac_r:.3f}  F1: {mac_f1:.3f}')

    # Step 15: Generate and save confusion matrix visualization
    # DATATYPE: PNG image file
    # WHY: Visual representation of per-class performance
    cm_path = Path(args.results_dir) / f'{model_name}_{mode}_{args.split}_confusion.png'
    plot_confusion_matrix(results['labels'], results['preds'],
                           class_names, cm_path, model_name=model_name)

    # Step 16: Save summary metrics as CSV for easy import into reports
    # DATATYPE: CSV file
    # WHY: Can load into Excel or use in future analysis/comparison
    import pandas as pd
    summary = pd.DataFrame([{
        'model':    model_name,
        'mode':     mode,
        'split':    args.split,
        'accuracy': round(results['accuracy'], 2),
        'loss':     round(results['loss'], 4),
        'fps':      round(results['fps'], 1),
        'macro_f1': round(mac_f1, 3),
    }])
    csv_out = Path(args.results_dir) / f'{model_name}_{mode}_{args.split}_summary.csv'
    summary.to_csv(csv_out, index=False)
    print(f'\n[INFO] Summary saved → {csv_out}')


if __name__ == '__main__':
    main()
