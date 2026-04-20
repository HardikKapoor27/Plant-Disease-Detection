"""
data_prep/download_dataset.py
──────────────────────────────
Downloads a small subset of the PlantVillage dataset from Kaggle
and organises it into train/ and valid/ folders.

Prerequisites:
  pip install kaggle
  Place your kaggle.json in ~/.kaggle/kaggle.json
  (Get it from: https://www.kaggle.com/settings → API → Create New Token)

Dataset used: abdallahalidev/plantvillage-dataset  (Kaggle)
Full dataset = ~2.5 GB / 38 classes / ~87K images
We use      = 14 classes / max 300 images each → ~30–50 MB unzipped
"""

import os
import sys
import shutil
import random
from pathlib import Path

# Add parent to path so we can import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, TRAIN_DIR, VALID_DIR,
    SELECTED_CLASSES, MAX_IMAGES_PER_CLASS, SEED
)

KAGGLE_DATASET  = "abdallahalidev/plantvillage-dataset"
ZIP_EXTRACT_DIR = os.path.join(DATA_DIR, "raw")
VALID_SPLIT     = 0.2   # 80% train, 20% validation

random.seed(SEED)


def download_from_kaggle():
    """Download the PlantVillage dataset via the Kaggle API (works with all versions)."""
    os.makedirs(ZIP_EXTRACT_DIR, exist_ok=True)
    print(f"📥 Downloading '{KAGGLE_DATASET}' from Kaggle …")

    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=ZIP_EXTRACT_DIR,
            unzip=True,
            quiet=False,
        )
        print("✅ Download complete.\n")
    except Exception as e:
        print(f"❌ Kaggle API error: {e}")
        raise

def find_source_dir(root: str) -> str:
    """
    The Kaggle zip nests images under different sub-folders depending on
    the version. Walk the extracted directory to find the folder that
    contains the class sub-directories.
    """
    for dirpath, dirnames, _ in os.walk(root):
        if any(cls in dirnames for cls in SELECTED_CLASSES):
            return dirpath
    raise FileNotFoundError(
        f"Could not find class folders inside {root}. "
        "Make sure the dataset downloaded correctly."
    )


def copy_subset(source_dir: str):
    """
    Copy a subset of images from source_dir into train/ and valid/ splits.
    Only the classes listed in SELECTED_CLASSES are included.
    """
    print(f"🔍 Source directory: {source_dir}")
    print(f"📦 Classes selected : {len(SELECTED_CLASSES)}")
    print(f"🖼️  Max images/class : {MAX_IMAGES_PER_CLASS}\n")

    total_train = 0
    total_valid = 0

    for cls in SELECTED_CLASSES:
        cls_src = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_src):
            print(f"  ⚠️  Skipping '{cls}' — folder not found.")
            continue

        images = [
            f for f in os.listdir(cls_src)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)

        # Cap to MAX_IMAGES_PER_CLASS
        if MAX_IMAGES_PER_CLASS:
            images = images[:MAX_IMAGES_PER_CLASS]

        split_idx   = int(len(images) * (1 - VALID_SPLIT))
        train_imgs  = images[:split_idx]
        valid_imgs  = images[split_idx:]

        # Copy train
        train_cls_dir = os.path.join(TRAIN_DIR, cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        for img in train_imgs:
            shutil.copy(os.path.join(cls_src, img),
                        os.path.join(train_cls_dir, img))

        # Copy valid
        valid_cls_dir = os.path.join(VALID_DIR, cls)
        os.makedirs(valid_cls_dir, exist_ok=True)
        for img in valid_imgs:
            shutil.copy(os.path.join(cls_src, img),
                        os.path.join(valid_cls_dir, img))

        print(f"  ✅ {cls:<45}  train={len(train_imgs):>3}  valid={len(valid_imgs):>3}")
        total_train += len(train_imgs)
        total_valid += len(valid_imgs)

    print(f"\n🎉 Dataset ready!")
    print(f"   Train images : {total_train}")
    print(f"   Valid images : {total_valid}")
    print(f"   Train dir    : {TRAIN_DIR}")
    print(f"   Valid dir    : {VALID_DIR}\n")


def main():
    # Step 1 – download if raw data not already present
    if not os.path.isdir(ZIP_EXTRACT_DIR) or not os.listdir(ZIP_EXTRACT_DIR):
        download_from_kaggle()
    else:
        print(f"📁 Raw data already present at {ZIP_EXTRACT_DIR}, skipping download.\n")

    # Step 2 – locate source class folders inside the extracted zip
    source_dir = find_source_dir(ZIP_EXTRACT_DIR)

    # Step 3 – build train/valid split
    copy_subset(source_dir)


if __name__ == "__main__":
    main()