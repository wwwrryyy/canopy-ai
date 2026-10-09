"""
autoencoder.py
==============
Convolutional Autoencoder (ANN) for unsupervised agricultural canopy baseline
reconstruction. Trained on healthy canopy only; reconstruction error highlights anomalies.

Single forward pass, designed for high-throughput TensorRT execution on edge hardware.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder.

    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = in_channels

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, H/2, W/2]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),           # [B, 32, H/4, W/4]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),           # bottleneck [B, 64, H/4, W/4]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 32, H/2, W/2]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 16, H, W]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Input shape: [Batch, Channels, Height, Width]
        Output shape: [Batch, Channels, Height, Width]
        """
        h, w = x.shape[-2:]
        pad_h = (4 - (h % 4)) % 4
        pad_w = (4 - (w % 4)) % 4

        if pad_h > 0 or pad_w > 0:
            x_padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            out = self.decoder(self.encoder(x_padded))
            return out[..., :h, :w]

        return self.decoder(self.encoder(x))
