"""
CS731 — Model Training Script
===============================
Trains one or more emotion recognition architectures and saves:
  - Best checkpoint per model (highest val accuracy)
  - Training curves CSV
  - Accuracy-vs-epoch plot (matches figures in both exemplar reports)

Usage
-----
  # Train one model
  python models/train.py --model swin_tiny --mode ekman6 --epochs 20

  # Train all comparison models (as done in Group 15's report)
  python models/train.py --all --mode ekman6 --epochs 20

  # ChatBox_V1 (Team 7 style)
  python models/train.py --model chatbox_v1 --mode ekman7 --epochs 30
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.emotion_dataset import get_loaders, mixup_data, mixup_criterion, cutmix_data
from models.model import build_model, AVAILABLE_MODELS

# ── Defaults (match exemplar configs) ─────────────────────────────────────────
DEFAULT_LR          = 1e-4
DEFAULT_BATCH        = 32
DEFAULT_EPOCHS       = 20
DEFAULT_IMG_SIZE     = 224
DEFAULT_WORKERS      = 4
DEFAULT_SPLITS_DIR   = 'data/splits'
DEFAULT_CHECKPOINTS  = 'models/checkpoints'
DEFAULT_RESULTS_DIR  = 'results/training'

# Models compared by Group 15
GROUP15_MODELS = [
    'efficientnet_b0', 'efficientnet_b3', 'swin_tiny',
    'mobilevit_s', 'convnext_tiny'
]


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'[INFO] Using GPU: {torch.cuda.get_device_name(0)}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('[INFO] Using Apple MPS')
    else:
        device = torch.device('cpu')
        print('[INFO] Using CPU (training will be slow)')
    return device


# ── One epoch ────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device,
                use_mixup: bool = False, mixup_alpha: float = 0.4,
                use_cutmix: bool = False) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy_%)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Optional augmentation: MixUp or CutMix ────────────────────────
        if use_mixup and np.random.rand() < 0.5:
            images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        elif use_cutmix and np.random.rand() < 0.5:
            images, y_a, y_b, lam = cutmix_data(images, labels, alpha=1.0)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)

        loss.backward()
        # Gradient clipping prevents exploding gradients (especially in Swin)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def val_epoch(model, loader, criterion, device) -> tuple[float, float]:
    """Run validation. Returns (avg_loss, accuracy_%)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(
    model_name:    str,
    mode:          str,
    splits_dir:    str,
    checkpoints_dir: str,
    results_dir:   str,
    epochs:        int   = DEFAULT_EPOCHS,
    lr:            float = DEFAULT_LR,
    batch_size:    int   = DEFAULT_BATCH,
    img_size:      int   = DEFAULT_IMG_SIZE,
    num_workers:   int   = DEFAULT_WORKERS,
    use_mixup:     bool  = True,
    use_cutmix:    bool  = True,
    pretrained:    bool  = True,
) -> dict:
    """
    Full training loop for one model.

    Returns a dict with epoch-by-epoch metrics for plotting.
    """
    device = get_device()

    # ── DataLoaders ──────────────────────────────────────────────────────────
    loaders = get_loaders(splits_dir, mode=mode, batch_size=batch_size,
                           img_size=img_size, num_workers=num_workers)
    if 'train' not in loaders or 'val' not in loaders:
        raise FileNotFoundError(
            f'Could not find {mode}_train.csv / {mode}_val.csv in {splits_dir}. '
            f'Run: python data/dataset_preparation.py --mode {mode}'
        )
    num_classes = loaders['train'].dataset.num_classes

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(model_name, num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)   # label smoothing helps
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ── Checkpoint and results paths ──────────────────────────────────────────
    ckpt_dir = Path(checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir  = Path(results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'{model_name}_{mode}_best.pt'
    csv_path  = res_dir  / f'{model_name}_{mode}_history.csv'

    # ── Training loop ─────────────────────────────────────────────────────────
    history     = []
    best_val_acc = 0.0
    best_epoch   = 0

    print(f'\n{"="*60}')
    print(f'  Training: {model_name}  |  Mode: {mode}  |  Epochs: {epochs}')
    print(f'  LR: {lr}  |  Batch: {batch_size}  |  Classes: {num_classes}')
    print(f'  MixUp: {use_mixup}  |  CutMix: {use_cutmix}')
    print(f'{"="*60}')

    # Write CSV header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc',
                          'val_loss', 'val_acc', 'lr'])

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()

        train_loss, train_acc = train_epoch(
            model, loaders['train'], optimizer, criterion, device,
            use_mixup=use_mixup, use_cutmix=use_cutmix
        )
        val_loss, val_acc = val_epoch(model, loaders['val'], criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        ep_time    = time.time() - ep_start

        row = {
            'epoch': epoch, 'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 2), 'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 2), 'lr': round(current_lr, 8)
        }
        history.append(row)

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                'epoch':       epoch,
                'model_name':  model_name,
                'mode':        mode,
                'num_classes': num_classes,
                'val_acc':     val_acc,
                'state_dict':  model.state_dict(),
                'optimizer':   optimizer.state_dict(),
            }, ckpt_path)
            star = ' ★ BEST'
        else:
            star = ''

        print(f'Epoch {epoch:3d}/{epochs} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | '
              f'{ep_time:.0f}s{star}')

        # Append to CSV
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row.values())

    total_time = time.time() - start_time
    print(f'\n✅ Training complete in {total_time/60:.1f} min')
    print(f'   Best Val Acc: {best_val_acc:.2f}% at epoch {best_epoch}')
    print(f'   Checkpoint: {ckpt_path}')

    return {'model_name': model_name, 'history': history,
            'best_val_acc': best_val_acc, 'best_epoch': best_epoch}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_training_curves(all_results: list, save_path: str | Path,
                          metric: str = 'val_acc') -> None:
    """
    Reproduce the accuracy-vs-epoch chart from the Group 15 final report.
    Plots multiple models on the same axes.

    Args:
        all_results: list of dicts from train_model()
        save_path:   where to save the PNG
        metric:      'val_acc' | 'val_loss' | 'train_acc'
    """
    import seaborn as sns
    palette = sns.color_palette('tab10', len(all_results))

    fig, ax = plt.subplots(figsize=(12, 6))

    for result, color in zip(all_results, palette):
        epochs   = [r['epoch']   for r in result['history']]
        values   = [r[metric]    for r in result['history']]
        best_ep  = result['best_epoch']
        best_val = result['best_val_acc']
        label    = f"{result['model_name']} (best {best_val:.2f}%)"

        ax.plot(epochs, values, marker='o', markersize=4,
                linewidth=1.5, label=label, color=color)
        ax.axvline(x=best_ep, color=color, linestyle='--', alpha=0.3)

    y_label_map = {'val_acc': 'Validation Accuracy (%)',
                   'val_loss': 'Validation Loss',
                   'train_acc': 'Training Accuracy (%)'}
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(y_label_map.get(metric, metric), fontsize=12)
    ax.set_title(f'Validation Accuracy vs Epoch for Each Model', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'[INFO] Saved training curves → {save_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='CS731 Model Training')
    p.add_argument('--model',      type=str,  default='swin_tiny',
                   help=f'Model name. Choices: {AVAILABLE_MODELS}')
    p.add_argument('--all',        action='store_true',
                   help='Train all Group 15 comparison models')
    p.add_argument('--mode',       type=str,  default='ekman6',
                   choices=['ekman6', 'ekman7', 'all8'])
    p.add_argument('--epochs',     type=int,  default=DEFAULT_EPOCHS)
    p.add_argument('--lr',         type=float,default=DEFAULT_LR)
    p.add_argument('--batch_size', type=int,  default=DEFAULT_BATCH)
    p.add_argument('--img_size',   type=int,  default=DEFAULT_IMG_SIZE)
    p.add_argument('--num_workers',type=int,  default=DEFAULT_WORKERS)
    p.add_argument('--splits_dir', type=str,  default=DEFAULT_SPLITS_DIR)
    p.add_argument('--ckpt_dir',   type=str,  default=DEFAULT_CHECKPOINTS)
    p.add_argument('--results_dir',type=str,  default=DEFAULT_RESULTS_DIR)
    p.add_argument('--no_mixup',   action='store_true', help='Disable MixUp')
    p.add_argument('--no_cutmix',  action='store_true', help='Disable CutMix')
    p.add_argument('--no_pretrain',action='store_true', help='Random init (no ImageNet)')
    return p.parse_args()


def main():
    args = parse_args()
    models_to_train = GROUP15_MODELS if args.all else [args.model]

    all_results = []
    for model_name in models_to_train:
        result = train_model(
            model_name    = model_name,
            mode          = args.mode,
            splits_dir    = args.splits_dir,
            checkpoints_dir = args.ckpt_dir,
            results_dir   = args.results_dir,
            epochs        = args.epochs,
            lr            = args.lr,
            batch_size    = args.batch_size,
            img_size      = args.img_size,
            num_workers   = args.num_workers,
            use_mixup     = not args.no_mixup,
            use_cutmix    = not args.no_cutmix,
            pretrained    = not args.no_pretrain,
        )
        all_results.append(result)

    # Plot comparison curves (replicates Group 15 Fig. 4 & 5)
    if len(all_results) > 0:
        plot_path = Path(args.results_dir) / f'training_curves_{args.mode}.png'
        plot_training_curves(all_results, save_path=plot_path)

    # Summary table
    print('\n── Final Summary ──────────────────────────────────────')
    print(f'{"Model":<35} {"Best Val Acc":>12} {"Best Epoch":>10}')
    print('─' * 60)
    for r in sorted(all_results, key=lambda x: x['best_val_acc'], reverse=True):
        print(f'{r["model_name"]:<35} {r["best_val_acc"]:>11.2f}% {r["best_epoch"]:>10}')


if __name__ == '__main__':
    main()
