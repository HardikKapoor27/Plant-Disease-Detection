"""
utils/evaluate.py
──────────────────
Evaluate a trained model on the validation set.

Outputs:
  • Overall accuracy
  • Per-class precision, recall, F1 (classification_report)
  • Confusion matrix heatmap saved to plots/confusion_matrix.png
"""

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_SAVE_PATH, PLOT_DIR, NUM_CLASSES
from models.resnet9 import ResNet9
from data_prep.data_loader import get_dataloaders, get_device


# ─── Inference over the validation set ───────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, save: bool = True):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalise

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150)
        print(f"📊 Confusion matrix saved → {path}")
    plt.show()


# ─── Main evaluation ──────────────────────────────────────────────────────────

def evaluate():
    device = get_device()
    _, valid_loader, class_names = get_dataloaders()

    # Load model
    model = ResNet9(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    print(f"✅ Model loaded from {MODEL_SAVE_PATH}\n")

    y_true, y_pred = get_predictions(model, valid_loader, device)

    # Overall accuracy
    accuracy = (y_true == y_pred).mean() * 100
    print(f"✅ Validation Accuracy : {accuracy:.2f}%\n")

    # Per-class report
    print("Classification Report:")
    print("─" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)


if __name__ == "__main__":
    evaluate()
