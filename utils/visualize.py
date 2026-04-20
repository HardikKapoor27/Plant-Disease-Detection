"""
utils/visualize.py
───────────────────
Plotting utilities:
  1. plot_training_history  – loss & accuracy curves
  2. show_sample_images     – grid of dataset samples
  3. plot_lr_schedule       – learning rate over batches
"""

import os
import sys
import pickle
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))
from config import PLOT_DIR, HISTORY_SAVE_PATH


# ─── 1. Training history plots ────────────────────────────────────────────────

def plot_training_history(history: dict = None, save: bool = True):
    """
    Plot training & validation loss/accuracy curves side by side.

    Args:
        history : dict with keys train_loss, train_acc, val_loss, val_acc
                  If None, loads from HISTORY_SAVE_PATH.
        save    : whether to save the figure as a PNG
    """
    if history is None:
        with open(HISTORY_SAVE_PATH, "rb") as f:
            history = pickle.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=15, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", markersize=4, label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]],
                 "b-o", markersize=4, label="Train Acc")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]],
                 "r-o", markersize=4, label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "training_history.png")
        plt.savefig(path, dpi=150)
        print(f"📈 Training history plot saved → {path}")
    plt.show()


# ─── 2. Sample images grid ────────────────────────────────────────────────────

def show_sample_images(dataloader, class_names: list,
                       num_images: int = 16, save: bool = True):
    """
    Display a grid of sample images from a DataLoader.

    Args:
        dataloader  : a PyTorch DataLoader
        class_names : list of class name strings
        num_images  : total images to display (will be rounded to a square)
        save        : whether to save the figure as a PNG
    """
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]

    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images_dn = (images * std + mean).clamp(0, 1)

    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle("Sample Training Images", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, (img, lbl) in enumerate(zip(images_dn, labels)):
        axes[i].imshow(img.permute(1, 2, 0).numpy())
        axes[i].set_title(class_names[lbl.item()].replace("___", "\n").replace("_", " "),
                          fontsize=7)
        axes[i].axis("off")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "sample_images.png")
        plt.savefig(path, dpi=150)
        print(f"🖼️  Sample images saved → {path}")
    plt.show()


# ─── 3. Learning rate schedule plot ──────────────────────────────────────────

def plot_lr_schedule(history: dict = None, save: bool = True):
    """Plot how the learning rate evolved during training."""
    if history is None:
        with open(HISTORY_SAVE_PATH, "rb") as f:
            history = pickle.load(f)

    lrs = history.get("lrs", [])
    if not lrs:
        print("No LR data found in history.")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(lrs, color="purple")
    plt.title("Learning Rate Schedule (OneCycleLR)", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "lr_schedule.png")
        plt.savefig(path, dpi=150)
        print(f"📉 LR schedule plot saved → {path}")
    plt.show()


# ─── CLI usage ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_training_history()
    plot_lr_schedule()
