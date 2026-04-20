"""
utils/predict.py
─────────────────
Utility for predicting the disease class of a single leaf image.

Usage (from command line):
    python utils/predict.py --image path/to/leaf.jpg

Usage (as a module):
    from utils.predict import predict_image
    label, confidence, top5 = predict_image("leaf.jpg")
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_SAVE_PATH, IMAGE_SIZE, NUM_CLASSES, CLASS_DISPLAY_NAMES
from models.resnet9 import ResNet9
from data_prep.data_loader import get_dataloaders, get_device

# ─── Transform for inference (no augmentation) ────────────────────────────────

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])


def load_model(device):
    """Load the trained EfficientNet-B0 model from disk."""
    import torchvision.models as tv_models
    import torch.nn as nn
    model = tv_models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def predict_image(image_path: str,
                  model=None,
                  device=None,
                  class_names=None,
                  top_k: int = 5):
    """
    Predict the disease class for a single leaf image.

    Args:
        image_path  : path to the image file
        model       : pre-loaded model (loaded once if None)
        device      : torch device
        class_names : list of class names from ImageFolder
        top_k       : number of top predictions to return

    Returns:
        predicted_label (str), confidence (float 0-1), top_k_preds (list of (label, conf))
    """
    if device is None:
        device = get_device()
    if model is None:
        model = load_model(device)
    if class_names is None:
        _, _, class_names = get_dataloaders()

    # Load & preprocess image
    img = Image.open(image_path).convert("RGB")
    tensor = INFER_TRANSFORM(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    # Forward pass
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

    # Top-k predictions
    topk_probs, topk_indices = torch.topk(probs, k=min(top_k, len(class_names)))
    top_k_preds = [
        (CLASS_DISPLAY_NAMES.get(class_names[idx.item()], class_names[idx.item()]),
         prob.item())
        for idx, prob in zip(topk_indices, topk_probs)
    ]

    best_class_idx  = topk_indices[0].item()
    best_class_name = class_names[best_class_idx]
    best_label      = CLASS_DISPLAY_NAMES.get(best_class_name, best_class_name)
    confidence      = topk_probs[0].item()

    return best_label, confidence, top_k_preds


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict plant disease from a leaf image")
    parser.add_argument("--image", required=True, help="Path to the leaf image")
    parser.add_argument("--top_k", type=int, default=5, help="Show top-k predictions")
    args = parser.parse_args()

    device      = get_device()
    model       = load_model(device)
    _, _, class_names = get_dataloaders()

    label, conf, top_k = predict_image(
        args.image, model=model, device=device,
        class_names=class_names, top_k=args.top_k
    )

    print(f"\n🌿 Predicted : {label}")
    print(f"   Confidence: {conf*100:.1f}%\n")
    print(f"Top-{args.top_k} predictions:")
    for rank, (lbl, p) in enumerate(top_k, 1):
        bar = "█" * int(p * 30)
        print(f"  {rank}. {lbl:<40} {p*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
