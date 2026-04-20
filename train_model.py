"""
train_model.py
───────────────
Master script to run the full training pipeline:

  1. Load data (train + valid DataLoaders)
  2. Initialize ResNet9 model
  3. Train with OneCycleLR for NUM_EPOCHS
  4. Plot training history
  5. Evaluate on validation set (classification report + confusion matrix)

Usage:
    python train_model.py
    python train_model.py --epochs 10   # quick test run
"""

import argparse
import random
import numpy as np
import torch
import sys

from config import NUM_EPOCHS, SEED, NUM_CLASSES
from data_prep.data_loader import get_dataloaders, get_device
from models.resnet9 import ResNet9
from models.train import train
from utils.visualize import plot_training_history, show_sample_images, plot_lr_schedule
from utils.evaluate import evaluate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(epochs: int):
    set_seed(SEED)

    print("=" * 60)
    print("  🌿 Plant Disease Detection — Training Pipeline")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset …")
    train_loader, valid_loader, class_names = get_dataloaders()
    print(f"  Classes     : {len(class_names)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Valid batches: {len(valid_loader)}")

    device = get_device()

    # ── 2. Show sample images ─────────────────────────────────────────────────
    print("\n[2/5] Saving sample images …")
    show_sample_images(train_loader, class_names, num_images=16)

    # ── 3. Build model ────────────────────────────────────────────────────────
    print("\n[3/5] Building ResNet9 model …")
    model = ResNet9(in_channels=3, num_classes=NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print(f"\n[4/5] Training for {epochs} epoch(s) …")
    history = train(model, train_loader, valid_loader, device, epochs=epochs)

    # ── 5. Plot & Evaluate ────────────────────────────────────────────────────
    print("\n[5/5] Plotting & evaluating …")
    plot_training_history(history)
    plot_lr_schedule(history)
    evaluate()

    print("\n✅ All done! Check the `plots/` and `saved_models/` directories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})"
    )
    args = parser.parse_args()
    main(args.epochs)
