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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.emotion_dataset import EmotionDataset, get_transforms
from models.model import build_model
from torch.utils.data import DataLoader


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str | Path, device: torch.device):
    """Load a training checkpoint and reconstruct the model."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name  = ckpt['model_name']
    num_classes = ckpt['num_classes']
    epoch       = ckpt['epoch']
    val_acc     = ckpt.get('val_acc', '?')

    print(f'[INFO] Loading checkpoint: {ckpt_path}')
    print(f'       Model: {model_name}  |  Classes: {num_classes}  |  '
          f'Epoch: {epoch}  |  Val Acc: {val_acc:.2f}%')

    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()
    return model, ckpt


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device, criterion: nn.Module) -> dict:
    """
    Full evaluation pass. Returns dict with all metrics.
    """
    all_preds  = []
    all_labels = []
    total_loss = 0.0
    total_time = 0.0
    n_batches  = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        t0      = time.perf_counter()
        outputs = model(images)
        total_time += (time.perf_counter() - t0)

        loss        = criterion(outputs, labels)
        total_loss += loss.item()
        n_batches  += 1

        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / n_batches
    accuracy = 100.0 * (all_preds == all_labels).mean()
    fps      = len(all_labels) / total_time

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
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names))))

    # Build annotation strings: "count\n(pct%)"
    annot = np.empty_like(cm, dtype=object)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            annot[i, j] = f'{cm[i,j]}\n{cm_norm[i,j]:.1f}%'

    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=100, ax=ax, linewidths=0.5, linecolor='gray')

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    title = f'Confusion Matrix on {"Test" if "test" in str(save_path) else "Validation"} Set'
    if model_name:
        title = f'{model_name} — {title}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'[INFO] Confusion matrix saved → {save_path}')


# ── Per-class metrics table ───────────────────────────────────────────────────

def print_classification_report(labels: np.ndarray, preds: np.ndarray,
                                  class_names: list) -> None:
    """Print sklearn classification report (precision, recall, F1 per class)."""
    report = classification_report(
        labels, preds,
        target_names=class_names,
        digits=3,
        zero_division=0
    )
    print('\n── Per-Class Metrics ──────────────────────────────────')
    print(report)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='CS731 Model Evaluation')
    p.add_argument('--checkpoint',  type=str, required=True,
                   help='Path to .pt checkpoint file')
    p.add_argument('--splits_dir',  type=str, default='data/splits',
                   help='Directory with split CSVs')
    p.add_argument('--split',       type=str, default='val',
                   choices=['train', 'val', 'test'])
    p.add_argument('--batch_size',  type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--results_dir', type=str, default='results/evaluation')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Load model from checkpoint
    model, ckpt = load_checkpoint(args.checkpoint, device)
    model_name  = ckpt['model_name']
    mode        = ckpt['mode']

    # DataLoader for the requested split
    csv_path = Path(args.splits_dir) / f'{mode}_{args.split}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f'Split CSV not found: {csv_path}\n'
            f'Run: python data/dataset_preparation.py --mode {mode}'
        )

    transform = get_transforms('val')   # always use val transforms for evaluation
    dataset   = EmotionDataset(csv_path, transform=transform)
    loader    = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers,
                           pin_memory=True)

    class_names = dataset.classes
    print(f'\n[INFO] Evaluating on {args.split} split '
          f'({len(dataset)} images, {len(class_names)} classes)')

    # Run evaluation
    criterion = nn.CrossEntropyLoss()
    results   = evaluate(model, loader, device, criterion)

    # Print summary
    print(f'\n── Results: {model_name} ({args.split}) ──────────────────')
    print(f'  Accuracy : {results["accuracy"]:.2f}%')
    print(f'  Loss     : {results["loss"]:.4f}')
    print(f'  Speed    : {results["fps"]:.1f} FPS')

    # Per-class report
    print_classification_report(results['labels'], results['preds'], class_names)

    # Macro averages
    mac_p  = precision_score(results['labels'], results['preds'], average='macro', zero_division=0)
    mac_r  = recall_score   (results['labels'], results['preds'], average='macro', zero_division=0)
    mac_f1 = f1_score       (results['labels'], results['preds'], average='macro', zero_division=0)
    print(f'Macro Precision: {mac_p:.3f}  Recall: {mac_r:.3f}  F1: {mac_f1:.3f}')

    # Confusion matrix
    cm_path = Path(args.results_dir) / f'{model_name}_{mode}_{args.split}_confusion.png'
    plot_confusion_matrix(results['labels'], results['preds'],
                           class_names, cm_path, model_name=model_name)

    # Save summary CSV
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
