"""
classifier.py
=============
Stage 3: Agricultural anomaly classifier (MobileNetV3-Small, ANN).

Classifies 64x64 crops flagged by Stage 2 into 5 classes. Includes a learnable
temperature parameter for calibrated probabilities (fit on a validation set).
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

CLASS_NAMES = [
    "Healthy Canopy",
    "Fungal/Bacterial Disease",
    "Pest Infestation",
    "Water Stress",
    "Weed Invasion",
]


class AgriculturalClassifier(nn.Module):
    """
    MobileNetV3-Small classifier for 64x64 anomaly crops.

    Args:
        in_channels: Input channels (3 for RGB; up to 7 with NIR/RedEdge/NDVI/NDRE).
        num_classes: Number of output classes.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.net = mobilenet_v3_small(weights=None, num_classes=num_classes)

        if in_channels != 3:
            first = self.net.features[0][0]
            self.net.features[0][0] = nn.Conv2d(
                in_channels, first.out_channels,
                kernel_size=first.kernel_size, stride=first.stride,
                padding=first.padding, bias=False,
            )

        # Temperature for post-hoc calibration; 1.0 = uncalibrated logits.
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns temperature-scaled logits [B, num_classes]."""
        return self.net(x) / self.temperature

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns calibrated class probabilities [B, num_classes]."""
        return torch.softmax(self.forward(x), dim=1)
