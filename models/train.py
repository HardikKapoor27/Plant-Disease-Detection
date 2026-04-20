"""
models/train.py
────────────────
Training loop for the Plant Disease Detection model.

Features:
  • OneCycleLR learning-rate schedule (fast convergence)
  • Per-epoch train & validation loss/accuracy tracking
  • Best-model checkpointing (saves the epoch with highest val accuracy)
  • tqdm progress bars
  • History dict returned for plotting
"""

import os
import sys
import time
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    NUM_EPOCHS, LEARNING_RATE, MAX_LR, WEIGHT_DECAY,
    MODEL_SAVE_PATH, HISTORY_SAVE_PATH
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy over a batch."""
    _, preds = torch.max(outputs, dim=1)
    return (preds == labels).float().mean().item()


# ─── Single epoch functions ───────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    running_acc  = 0.0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_acc  += accuracy(outputs, labels)

    n = len(loader)
    return running_loss / n, running_acc / n


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc  = 0.0

    for images, labels in tqdm(loader, desc="  Valid", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item()
        running_acc  += accuracy(outputs, labels)

    n = len(loader)
    return running_loss / n, running_acc / n


# ─── Full training pipeline ───────────────────────────────────────────────────

def train(model, train_loader, valid_loader, device,
          epochs: int = NUM_EPOCHS, max_lr: float = MAX_LR) -> dict:
    """
    Train `model` for `epochs` epochs.

    Returns:
        history dict with keys:
          train_loss, train_acc, val_loss, val_acc, lrs
    """
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE,
                     weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lrs":        [],
    }

    best_val_acc  = 0.0
    best_epoch    = 0
    start_time    = time.time()

    print(f"\n{'─'*60}")
    print(f"  Training for {epochs} epochs on {device}")
    print(f"{'─'*60}")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch [{epoch:02d}/{epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_acc = validate_one_epoch(
            model, valid_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lrs"].append(scheduler.get_last_lr()[0])

        print(
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  💾 Best model saved (val_acc={best_val_acc:.4f})")

    elapsed = time.time() - start_time
    print(f"\n{'─'*60}")
    print(f"  Training complete in {elapsed/60:.1f} min")
    print(f"  Best val_acc = {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"  Model saved  → {MODEL_SAVE_PATH}")
    print(f"{'─'*60}\n")

    # Persist history for later plotting
    with open(HISTORY_SAVE_PATH, "wb") as f:
        pickle.dump(history, f)

    return history
