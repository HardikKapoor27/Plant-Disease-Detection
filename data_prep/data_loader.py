"""
data_prep/data_loader.py
─────────────────────────
Builds PyTorch DataLoaders for the plant disease dataset.
Includes:
  - Training augmentations (flip, rotate, colour jitter, random crop)
  - Validation transforms (resize + centre crop only)
  - Class-index to class-name mapping
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    TRAIN_DIR, VALID_DIR, IMAGE_SIZE, BATCH_SIZE, SEED
)

# ─── Transforms ───────────────────────────────────────────────────────────────

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),  # slightly larger for random crop
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats
        std =[0.229, 0.224, 0.225]
    ),
])

VALID_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])


# ─── Dataset & DataLoader builders ───────────────────────────────────────────

def get_datasets():
    """Return (train_dataset, valid_dataset) as ImageFolder objects."""
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=TRAIN_TRANSFORMS)
    valid_ds = datasets.ImageFolder(VALID_DIR, transform=VALID_TRANSFORMS)
    return train_ds, valid_ds


def get_dataloaders(num_workers: int = 2):
    """
    Return (train_loader, valid_loader, class_names).

    Args:
        num_workers: parallel workers for DataLoader; set to 0 on Windows.
    """
    train_ds, valid_ds = get_datasets()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = train_ds.classes
    return train_loader, valid_loader, class_names


def get_device():
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"🖥️  Using device: {device}")
    return device


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, valid_loader, class_names = get_dataloaders()
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Valid batches : {len(valid_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Batch shape   : {images.shape}   Labels: {labels.shape}")
