"""
extract_and_fix.py
───────────────────
Extracts the plantvillage-dataset.zip and organizes into train/valid.
Run this once from the project root.
"""

import os
import sys
import zipfile
import shutil
import random
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from config import (
    DATA_DIR, TRAIN_DIR, VALID_DIR,
    SELECTED_CLASSES, MAX_IMAGES_PER_CLASS, SEED
)

random.seed(SEED)
VALID_SPLIT = 0.2
RAW_DIR     = os.path.join(DATA_DIR, "raw")
ZIP_PATH    = os.path.join(RAW_DIR, "plantvillage-dataset.zip")


# ── Step 1: Extract zip ───────────────────────────────────────────────────────

def extract_zip():
    if not os.path.exists(ZIP_PATH):
        print(f"❌ ZIP not found at: {ZIP_PATH}")
        sys.exit(1)

    extract_to = os.path.join(RAW_DIR, "extracted")
    if os.path.isdir(extract_to) and os.listdir(extract_to):
        print(f"📁 Already extracted at: {extract_to}\n")
        return extract_to

    print(f"📦 Extracting ZIP (~2.5 GB, please wait) ...")
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        members = z.namelist()
        total   = len(members)
        for i, member in enumerate(members, 1):
            z.extract(member, extract_to)
            if i % 2000 == 0:
                pct = i / total * 100
                print(f"   {pct:.0f}% ... ({i}/{total} files)")

    print(f"✅ Extraction complete → {extract_to}\n")
    return extract_to


# ── Step 2: Find class folders ────────────────────────────────────────────────

def find_class_folders(root):
    """Walk root to find the directory containing class sub-folders."""
    print(f"🔍 Scanning extracted folder ...")
    for dirpath, dirnames, filenames in os.walk(root):
        # Look for a folder containing plant class sub-folders
        img_dirs = [
            d for d in dirnames
            if os.path.isdir(os.path.join(dirpath, d)) and
            any(f.lower().endswith((".jpg", ".jpeg", ".png"))
                for f in os.listdir(os.path.join(dirpath, d))[:5])
        ]
        if len(img_dirs) >= 5:
            print(f"✅ Class folders found in: {dirpath}")
            print(f"   Sample classes: {img_dirs[:5]}")
            return dirpath

    # Fallback: find deepest folder with images
    for dirpath, dirnames, filenames in os.walk(root):
        images = [f for f in filenames if f.lower().endswith((".jpg",".jpeg",".png"))]
        if len(images) > 100:
            parent = str(Path(dirpath).parent)
            print(f"✅ Found image-heavy folder, using parent: {parent}")
            return parent

    return None


# ── Step 3: Copy subset into train/valid ─────────────────────────────────────

def copy_subset(source_dir):
    available = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ])

    print(f"\n📋 Available classes ({len(available)}):")
    for c in available:
        print(f"   {c}")

    # Match selected classes (flexible matching)
    to_copy = []
    for sel in SELECTED_CLASSES:
        if sel in available:
            to_copy.append((sel, sel))
        else:
            # Flexible: strip underscores and compare lowercase
            sel_key = sel.lower().replace("_","").replace(" ","")
            match = next(
                (a for a in available
                 if a.lower().replace("_","").replace(" ","") == sel_key),
                None
            )
            if match:
                to_copy.append((match, sel))
                print(f"   ✔ Mapped '{match}' → '{sel}'")
            else:
                print(f"   ⚠ Could not find match for '{sel}'")

    if not to_copy:
        print("\n⚠️  No exact matches found — using ALL classes.")
        to_copy = [(c, c) for c in available]

    print(f"\n🚀 Copying {len(to_copy)} classes → train/valid ...\n")

    total_train = total_valid = 0
    for src_name, dst_name in to_copy:
        src_path = os.path.join(source_dir, src_name)
        imgs = [
            f for f in os.listdir(src_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(imgs)
        if MAX_IMAGES_PER_CLASS:
            imgs = imgs[:MAX_IMAGES_PER_CLASS]

        split      = int(len(imgs) * (1 - VALID_SPLIT))
        train_imgs = imgs[:split]
        valid_imgs = imgs[split:]

        os.makedirs(os.path.join(TRAIN_DIR, dst_name), exist_ok=True)
        os.makedirs(os.path.join(VALID_DIR, dst_name), exist_ok=True)

        for f in train_imgs:
            shutil.copy(os.path.join(src_path, f),
                        os.path.join(TRAIN_DIR, dst_name, f))
        for f in valid_imgs:
            shutil.copy(os.path.join(src_path, f),
                        os.path.join(VALID_DIR, dst_name, f))

        print(f"  ✅ {dst_name:<45} train={len(train_imgs):>3}  valid={len(valid_imgs):>3}")
        total_train += len(train_imgs)
        total_valid += len(valid_imgs)

    print(f"\n🎉 Done!")
    print(f"   Train : {total_train} images")
    print(f"   Valid : {total_valid} images")
    print(f"\n▶  Now run:  python train_model.py\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Check if already organised
    if os.path.isdir(TRAIN_DIR) and os.listdir(TRAIN_DIR):
        print(f"✅ Train folder already exists. Running train_model.py directly.")
        print(f"   Run: python train_model.py")
        return

    extract_dir  = extract_zip()
    source_dir   = find_class_folders(extract_dir)

    if not source_dir:
        # Maybe the "plantvillage dataset" folder already has class dirs
        alt = os.path.join(RAW_DIR, "plantvillage dataset")
        if os.path.isdir(alt):
            source_dir = find_class_folders(alt)

    if not source_dir:
        print("❌ Could not find class image folders.")
        print("   Run 'dir data\\raw\\extracted' and paste the output.")
        sys.exit(1)

    copy_subset(source_dir)


if __name__ == "__main__":
    main()
