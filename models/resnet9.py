"""
models/resnet9.py
──────────────────
ResNet9 — a compact residual CNN suited for medium-sized image datasets.
Much faster to train than standard ResNet-18/34/50, yet achieves excellent
accuracy on datasets like PlantVillage.

Architecture overview:
  Conv Block 1  → 64 filters
  Conv Block 2  → 128 filters  + MaxPool
  Residual Block 1 (128 filters)
  Conv Block 3  → 256 filters  + MaxPool
  Conv Block 4  → 512 filters  + MaxPool
  Residual Block 2 (512 filters)
  Classifier head (Adaptive Avg Pool → Flatten → Linear → Dropout → Linear)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import NUM_CLASSES


def conv_block(in_channels: int, out_channels: int,
               pool: bool = False) -> nn.Sequential:
    """
    Standard conv block: Conv2d → BatchNorm → ReLU → (optional MaxPool).
    """
    layers = [
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    """
    Lightweight 9-layer residual network.

    Args:
        in_channels : number of input channels (3 for RGB images)
        num_classes : number of output classes
    """

    def __init__(self, in_channels: int = 3, num_classes: int = NUM_CLASSES):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────────
        self.conv1 = conv_block(in_channels, 64)            # 224 → 224
        self.conv2 = conv_block(64, 128, pool=True)         # 224 → 112

        # ── Residual block 1 ──────────────────────────────────────────────────
        self.res1 = nn.Sequential(
            conv_block(128, 128),
            conv_block(128, 128),
        )

        # ── Down-sampling blocks ───────────────────────────────────────────────
        self.conv3 = conv_block(128, 256, pool=True)        # 112 → 56
        self.conv4 = conv_block(256, 512, pool=True)        # 56  → 28

        # ── Residual block 2 ──────────────────────────────────────────────────
        self.res2 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512),
        )

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # → (B, 512, 1, 1)
            nn.Flatten(),               # → (B, 512)
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.conv2(x)

        # Residual 1
        x = self.res1(x) + x

        # Down-sampling
        x = self.conv3(x)
        x = self.conv4(x)

        # Residual 2
        x = self.res2(x) + x

        # Head
        x = self.classifier(x)
        return x


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    model = ResNet9(in_channels=3, num_classes=NUM_CLASSES)
    dummy  = torch.randn(4, 3, 224, 224)
    output = model(dummy)
    print(f"Model output shape : {output.shape}")   # should be (4, 14)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters   : {total_params:,}")
