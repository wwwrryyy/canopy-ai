"""
model.py
========
Lightweight Convolutional U-Net (ANN) for aerial imagery despeckling and restoration.

Architecture: Single-pass encoder-decoder with skip connections, batch normalization,
and LeakyReLU activations. Designed for ultra-high throughput on edge hardware (Jetson Orin Nano).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> LeakyReLU block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransConvBlock(nn.Module):
    """ConvTranspose2d -> BatchNorm2d -> LeakyReLU block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DespecklerUNet(nn.Module):
    """Lightweight Convolutional U-Net for aerial image restoration.

    Encoder: 3 conv blocks with spatial downsampling (stride=2).
    Decoder: 2 transposed-conv blocks with skip connections (concatenation).
    Readout: 1x1 conv with Sigmoid activation for normalized [0, 1] radiance.
    Single forward pass.

    Args:
        in_channels: Number of input channels (1 for grayscale / SAR, 3 for RGB, 5-7 for multispectral).
        base_ch: Base channel count (doubled at each encoder stage).
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch, stride=2)       # [B, base, H/2, W/2]
        self.enc2 = ConvBlock(base_ch, base_ch * 2, stride=2)      # [B, base*2, H/4, W/4]
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, stride=1)  # bottleneck [B, base*4, H/4, W/4]

        # Decoder with skip connections
        self.dec1 = TransConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)  # [B, base*2, H/2, W/2]
        self.dec2 = TransConvBlock(base_ch * 2 + base_ch, base_ch)          # [B, base, H, W]

        # 1x1 conv readout with Sigmoid to clamp to [0, 1]
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (4 - (h % 4)) % 4
        pad_w = (4 - (w % 4)) % 4

        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d1 = self.dec1(torch.cat([e3, e2], dim=1))
        d2 = self.dec2(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d2)
        if pad_h > 0 or pad_w > 0:
            return out[..., :h, :w]
        return out
