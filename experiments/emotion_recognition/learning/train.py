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
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'application' / 'mock_programs'))
from emotion_dataset import get_loaders, mixup_data, mixup_criterion, cutmix_data
from model import build_model, AVAILABLE_MODELS

# ── Defaults (match exemplar configs) ─────────────────────────────────────────
# WHY: These defaults match the exemplar reports, making results directly comparable
DEFAULT_LR          = 1e-4
DEFAULT_BATCH        = 32
DEFAULT_EPOCHS       = 20
DEFAULT_IMG_SIZE     = 224
DEFAULT_WORKERS      = 4
DEFAULT_SPLITS_DIR   = 'data/splits'
DEFAULT_CHECKPOINTS  = 'models/checkpoints'
DEFAULT_RESULTS_DIR  = 'results/training'

# Models compared by Group 15 (exemplar team)
GROUP15_MODELS = [
    'efficientnet_b0', 'efficientnet_b3', 'swin_tiny',
    'mobilevit_s', 'convnext_tiny'
]


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Detect and return the best available compute device."""
    # Step 1: Check if NVIDIA GPU available
    # DATATYPE: torch.device
    # WHY: GPU training is 10-100x faster than CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'[INFO] Using GPU: {torch.cuda.get_device_name(0)}')

    # Step 2: Check if Apple Silicon GPU available (MacBook M1/M2/etc)
    # WHY: MPS (Metal Performance Shaders) is faster than CPU but slower than NVIDIA
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('[INFO] Using Apple MPS')

    # Step 3: Fall back to CPU
    else:
        device = torch.device('cpu')
        print('[INFO] Using CPU (training will be slow)')

    return device


# ── One epoch ────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device,
                use_mixup: bool = False, mixup_alpha: float = 0.4,
                use_cutmix: bool = False) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy_%)."""
    # Step 1: Set model to training mode (enables dropout, batch norm updates)
    # WHY: Batch norm and dropout behave differently in train vs eval mode
    model.train()

    # Step 2: Initialize accumulators for loss and accuracy
    # DATATYPE: floats and ints
    total_loss, correct, total = 0.0, 0, 0

    # Step 3: Iterate over mini-batches from DataLoader
    # DATATYPE: images is torch.Tensor (batch_size, 3, 224, 224)
    #           labels is torch.Tensor (batch_size,) with integer class ids
    for images, labels in loader:
        # Step 4: Move data to GPU (if available) without blocking CPU
        # WHY: non_blocking=True allows CPU to continue preparing next batch
        #      while GPU processes current batch (pipeline parallelism)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Optional augmentation: MixUp or CutMix ────────────────────────
        # WHY: These augmentations prevent overfitting by creating "soft labels"
        #      and forcing the model to learn multiple class boundaries per batch

        if use_mixup and np.random.rand() < 0.5:
            # Step 5A: Apply MixUp to 50% of batches
            # DATATYPE: mixed_x is same shape as images; y_a, y_b are label tensors; lam is float
            images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha)

            # Step 5B: Zero gradients from previous batch
            # WHY: PyTorch accumulates gradients by default; we want fresh gradients per batch
            optimizer.zero_grad()

            # Step 5C: Forward pass: model predicts logits for mixed images
            # DATATYPE: outputs is torch.Tensor (batch_size, num_classes) with raw scores
            outputs = model(images)

            # Step 5D: Compute blended loss (weighted combination of two class losses)
            # WHY: Mixed image with λ fraction of class A and (1-λ) of class B
            #      should have loss weighted similarly
            loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        elif use_cutmix and np.random.rand() < 0.5:
            # Step 5E: Apply CutMix to 50% of batches (if not using MixUp)
            # DATATYPE: same as MixUp
            images, y_a, y_b, lam = cutmix_data(images, labels, alpha=1.0)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        else:
            # Step 5F: Standard training without augmentation
            # WHY: Use regular cross-entropy loss when no augmentation applied

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            # DATATYPE: outputs is (batch_size, num_classes)
            outputs = model(images)

            # Compute loss: cross-entropy between predicted logits and true labels
            # DATATYPE: scalar tensor (single float value)
            loss    = criterion(outputs, labels)

        # Step 6: Backward pass: compute gradients
        # WHY: Automatic differentiation through the model
        loss.backward()

        # Step 7: Gradient clipping prevents exploding gradients
        # WHY: Very deep models like Swin can have unstable gradients
        #      Clipping ensures gradient norm stays ≤ 1.0
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step 8: Update model parameters using gradients
        # WHY: AdamW optimizer computes adaptive learning rates per parameter
        optimizer.step()

        # Step 9: Accumulate loss for averaging
        # DATATYPE: float (scalar)
        total_loss += loss.item()

        # Step 10: Compute predictions and count correct ones
        # DATATYPE: predicted is (batch_size,) with class indices
        # WHY: outputs.max(1) gets the highest logit per sample (argmax)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()  # Count correct predictions
        total   += labels.size(0)                      # Count total samples

    # Step 11: Compute average loss and accuracy for this epoch
    # DATATYPE: floats
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    return avg_loss, accuracy


@torch.no_grad()  # Disable gradient computation (validation doesn't need backprop)
def val_epoch(model, loader, criterion, device) -> tuple[float, float]:
    """Run validation. Returns (avg_loss, accuracy_%)."""
    # Step 1: Set model to eval mode (disables dropout, uses running batch norm stats)
    # WHY: Batch norm and dropout are non-deterministic in train mode
    model.eval()

    # Step 2: Initialize accumulators
    total_loss, correct, total = 0.0, 0, 0

    # Step 3: Iterate over validation batches
    for images, labels in loader:
        # Step 4: Move data to device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Step 5: Forward pass (no backprop because of @torch.no_grad() decorator)
        outputs = model(images)

        # Step 6: Compute validation loss
        loss    = criterion(outputs, labels)
        total_loss += loss.item()

        # Step 7: Count correct predictions
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    # Step 8: Compute average metrics for this validation epoch
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
    # Step 1: Get compute device (GPU, MPS, or CPU)
    device = get_device()

    # ── DataLoaders ──────────────────────────────────────────────────────────
    # Step 2: Load all data splits from CSV manifests
    # DATATYPE: dict mapping split names to DataLoader objects
    # WHY: DataLoader handles batching, shuffling, parallel loading
    loaders = get_loaders(splits_dir, mode=mode, batch_size=batch_size,
                           img_size=img_size, num_workers=num_workers)

    # Step 3: Verify train and val splits exist
    if 'train' not in loaders or 'val' not in loaders:
        raise FileNotFoundError(
            f'Could not find {mode}_train.csv / {mode}_val.csv in {splits_dir}. '
            f'Run: python data/dataset_preparation.py --mode {mode}'
        )

    # Step 4: Get number of classes from the dataset
    # DATATYPE: int (6 for ekman6, 7 for ekman7, 8 for all8)
    # WHY: Model output layer needs exactly num_classes neurons
    num_classes = loaders['train'].dataset.num_classes

    # ── Model ─────────────────────────────────────────────────────────────────
    # Step 5: Instantiate model architecture
    # DATATYPE: nn.Module
    # WHY: pretrained=True loads ImageNet weights (transfer learning);
    #      much faster training than random initialization
    model = build_model(model_name, num_classes=num_classes, pretrained=pretrained)

    # Step 6: Move model to device (GPU/CPU)
    # WHY: All computations must be on same device
    model = model.to(device)

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    # Step 7: Create loss function with label smoothing
    # DATATYPE: nn.CrossEntropyLoss
    # WHY: label_smoothing=0.1 prevents the model from becoming overconfident;
    #      improves generalization by regularizing the loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Step 8: Create optimizer (AdamW with weight decay)
    # DATATYPE: AdamW optimizer
    # WHY: AdamW adapts learning rates per parameter; weight_decay prevents overfitting
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Step 9: Create learning rate scheduler
    # DATATYPE: CosineAnnealingLR scheduler
    # WHY: Cosine annealing gradually decreases LR from initial to minimum value;
    #      eta_min ensures we don't go too low (0.1% of initial LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ── Checkpoint and results paths ──────────────────────────────────────────
    # Step 10: Create checkpoint directory and construct checkpoint path
    # DATATYPE: Path objects
    ckpt_dir = Path(checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Step 11: Create results directory for CSV and plots
    res_dir  = Path(results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Step 12: Define exact paths for best checkpoint and training history
    ckpt_path = ckpt_dir / f'{model_name}_{mode}_best.pt'
    csv_path  = res_dir  / f'{model_name}_{mode}_history.csv'

    # ── Training loop ─────────────────────────────────────────────────────────
    # Step 13: Initialize tracking for best validation accuracy
    # DATATYPE: float (starts at 0)
    # WHY: We only save checkpoint when validation accuracy improves
    history     = []
    best_val_acc = 0.0
    best_epoch   = 0

    # Step 14: Print training configuration
    print(f'\n{"="*60}')
    print(f'  Training: {model_name}  |  Mode: {mode}  |  Epochs: {epochs}')
    print(f'  LR: {lr}  |  Batch: {batch_size}  |  Classes: {num_classes}')
    print(f'  MixUp: {use_mixup}  |  CutMix: {use_cutmix}')
    print(f'{"="*60}')

    # Step 15: Write CSV header for training history
    # DATATYPE: CSV file with columns for epoch, losses, accuracies, learning rate
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc',
                          'val_loss', 'val_acc', 'lr'])

    # Step 16: Record wall-clock time at training start
    start_time = time.time()

    # Step 17: Main training loop (one iteration = one epoch)
    for epoch in range(1, epochs + 1):
        # Step 17A: Record epoch start time
        ep_start = time.time()

        # Step 17B: Run one training epoch
        # DATATYPE: floats (loss and accuracy percentage)
        train_loss, train_acc = train_epoch(
            model, loaders['train'], optimizer, criterion, device,
            use_mixup=use_mixup, use_cutmix=use_cutmix
        )

        # Step 17C: Run one validation epoch
        val_loss, val_acc = val_epoch(model, loaders['val'], criterion, device)

        # Step 17D: Step the learning rate scheduler
        # WHY: Adjusts learning rate for next epoch based on cosine schedule
        scheduler.step()

        # Step 17E: Get current learning rate for logging
        # DATATYPE: float (in scientific notation like 1e-4)
        current_lr = scheduler.get_last_lr()[0]

        # Step 17F: Compute epoch wall-clock time
        ep_time    = time.time() - ep_start

        # Step 17G: Store metrics for this epoch
        # DATATYPE: dict with keys: epoch, train_loss, train_acc, val_loss, val_acc, lr
        row = {
            'epoch': epoch, 'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 2), 'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 2), 'lr': round(current_lr, 8)
        }
        history.append(row)

        # Step 17H: Check if this is the best epoch so far
        # WHY: Only save checkpoint on improvement (prevents wasting disk space)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch

            # Step 17I: Save checkpoint with full model and optimizer state
            # DATATYPE: .pt (PyTorch checkpoint) file
            # WHY: Saving optimizer state allows resuming training from this checkpoint
            torch.save({
                'epoch':       epoch,
                'model_name':  model_name,
                'mode':        mode,
                'num_classes': num_classes,
                'val_acc':     val_acc,
                'state_dict':  model.state_dict(),  # Model weights
                'optimizer':   optimizer.state_dict(),  # Optimizer state
            }, ckpt_path)
            star = ' ★ BEST'
        else:
            star = ''

        # Step 17J: Print progress (one line per epoch)
        print(f'Epoch {epoch:3d}/{epochs} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | '
              f'{ep_time:.0f}s{star}')

        # Step 17K: Append this epoch's metrics to CSV
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row.values())

    # Step 18: Compute total training time
    total_time = time.time() - start_time

    # Step 19: Print training summary
    print(f'\n✅ Training complete in {total_time/60:.1f} min')
    print(f'   Best Val Acc: {best_val_acc:.2f}% at epoch {best_epoch}')
    print(f'   Checkpoint: {ckpt_path}')

    # Step 20: Return results dict for plotting
    # DATATYPE: dict with model metrics
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
    # Step 1: Import seaborn for color palettes
    import seaborn as sns

    # Step 2: Generate distinct colors for each model
    # DATATYPE: list of RGB tuples
    # WHY: Makes it easy to distinguish model curves visually
    palette = sns.color_palette('tab10', len(all_results))

    # Step 3: Create figure and axis for the plot
    # DATATYPE: matplotlib Figure and Axes objects
    fig, ax = plt.subplots(figsize=(12, 6))

    # Step 4: Plot one curve per model
    for result, color in zip(all_results, palette):
        # Step 4A: Extract epoch numbers and metric values from history
        # DATATYPE: list of ints and floats
        epochs   = [r['epoch']   for r in result['history']]
        values   = [r[metric]    for r in result['history']]

        # Step 4B: Get best epoch and value for legend
        best_ep  = result['best_epoch']
        best_val = result['best_val_acc']

        # Step 4C: Create informative label for legend
        # Example: "swin_tiny (best 89.45%)"
        label    = f"{result['model_name']} (best {best_val:.2f}%)"

        # Step 4D: Plot line for this model
        # WHY: marker='o' shows individual epochs; linewidth controls line thickness
        ax.plot(epochs, values, marker='o', markersize=4,
                linewidth=1.5, label=label, color=color)

        # Step 4E: Add vertical dashed line at best epoch
        # WHY: Visual indicator of where best performance occurred
        ax.axvline(x=best_ep, color=color, linestyle='--', alpha=0.3)

    # Step 5: Set axis labels and title
    # DATATYPE: strings
    y_label_map = {'val_acc': 'Validation Accuracy (%)',
                   'val_loss': 'Validation Loss',
                   'train_acc': 'Training Accuracy (%)'}
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(y_label_map.get(metric, metric), fontsize=12)
    ax.set_title(f'Validation Accuracy vs Epoch for Each Model', fontsize=14, fontweight='bold')

    # Step 6: Add legend and grid
    # WHY: Legend identifies which line is which model
    #      Grid makes it easier to read values
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Step 7: Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'[INFO] Saved training curves → {save_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description='CS731 Model Training')

    # WHY: User can choose which model to train
    p.add_argument('--model',      type=str,  default='swin_tiny',
                   help=f'Model name. Choices: {AVAILABLE_MODELS}')

    # WHY: --all trains all Group 15 models for comparison
    p.add_argument('--all',        action='store_true',
                   help='Train all Group 15 comparison models')

    # WHY: Different emotion class sets for different research questions
    p.add_argument('--mode',       type=str,  default='ekman6',
                   choices=['ekman6', 'ekman7', 'all8'])

    # WHY: More epochs = longer training but better convergence (up to a point)
    p.add_argument('--epochs',     type=int,  default=DEFAULT_EPOCHS)
    p.add_argument('--lr',         type=float,default=DEFAULT_LR)
    p.add_argument('--batch_size', type=int,  default=DEFAULT_BATCH)
    p.add_argument('--img_size',   type=int,  default=DEFAULT_IMG_SIZE)
    p.add_argument('--num_workers',type=int,  default=DEFAULT_WORKERS)
    p.add_argument('--splits_dir', type=str,  default=DEFAULT_SPLITS_DIR)
    p.add_argument('--ckpt_dir',   type=str,  default=DEFAULT_CHECKPOINTS)
    p.add_argument('--results_dir',type=str,  default=DEFAULT_RESULTS_DIR)

    # WHY: Ablation study: test with/without augmentation techniques
    p.add_argument('--no_mixup',   action='store_true', help='Disable MixUp')
    p.add_argument('--no_cutmix',  action='store_true', help='Disable CutMix')
    p.add_argument('--no_pretrain',action='store_true', help='Random init (no ImageNet)')

    return p.parse_args()


def main():
    # Step 1: Parse command-line arguments
    args = parse_args()

    # Step 2: Determine which models to train
    # DATATYPE: list of strings
    # WHY: Can train a single model or all comparison models
    models_to_train = GROUP15_MODELS if args.all else [args.model]

    # Step 3: Train each model and collect results
    # DATATYPE: list of dicts from train_model()
    all_results = []
    for model_name in models_to_train:
        # Step 3A: Train this model
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

    # Step 4: Plot comparison curves if multiple models trained
    # WHY: Visual comparison helps understand relative performance
    if len(all_results) > 0:
        plot_path = Path(args.results_dir) / f'training_curves_{args.mode}.png'
        plot_training_curves(all_results, save_path=plot_path)

    # Step 5: Print final summary table (sorted by best accuracy)
    # WHY: Easy-to-read format for comparing models
    print('\n── Final Summary ──────────────────────────────────────')
    print(f'{"Model":<35} {"Best Val Acc":>12} {"Best Epoch":>10}')
    print('─' * 60)
    for r in sorted(all_results, key=lambda x: x['best_val_acc'], reverse=True):
        print(f'{r["model_name"]:<35} {r["best_val_acc"]:>11.2f}% {r["best_epoch"]:>10}')


if __name__ == '__main__':
    main()
