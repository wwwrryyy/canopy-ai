"""
train_autoencoder.py
====================
Unsupervised training of the Convolutional Autoencoder (Stage 2) on healthy canopy
patches, plus ONNX export for edge deployment (e.g., NVIDIA Jetson Orin Nano).
"""

import os
import sys
import glob
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from autoencoder import ConvAutoencoder

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


class PatchDataset(Dataset):
    """Dataset to load extracted patches from disk."""
    def __init__(self, data_dir: str):
        self.paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        if not self.paths:
            raise FileNotFoundError(f"No patches found in {data_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load patch {self.paths[idx]}")
        img = (img.astype(np.float32) / 255.0)
        return torch.from_numpy(img).unsqueeze(0)


def generate_synthetic_terrain_batch(batch_size=16, height=128, width=128, in_channels=1):
    """
    Generates synthetic terrain baselines with textures, gradients, and mild noise
    for unsupervised training when no real patches are available.
    """
    batch = []
    for _ in range(batch_size):
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)

        freq = np.random.uniform(2.0, 8.0)
        terrain = 0.5 + 0.3 * np.sin(freq * xx) * np.cos(freq * yy)

        texture = np.random.normal(0.0, 0.05, (height, width))
        frame = np.clip(terrain + texture, 0.0, 1.0).astype(np.float32)

        if in_channels == 3:
            frame = np.stack([frame, frame * 0.9, frame * 1.1], axis=0)
        else:
            frame = np.expand_dims(frame, axis=0)

        batch.append(frame)

    return torch.tensor(np.array(batch), dtype=torch.float32)


def train_and_export(data_dir: str = None, epochs: int = 15, batch_size: int = 16, lr: float = 1e-3):
    print("[+] Starting Convolutional Autoencoder Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")

    in_channels = 1
    h, w = 128, 128

    loader = None
    if data_dir and os.path.isdir(data_dir):
        try:
            dataset = PatchDataset(data_dir)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=(len(dataset) >= batch_size),
            )
            print(f"[+] Loaded real dataset: {len(dataset)} samples from {data_dir}")
        except FileNotFoundError:
            print(f"[-] No patches found in {data_dir}, using synthetic fallback.")

    model = ConvAutoencoder(in_channels=in_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0

        if loader is not None:
            for batch_inputs in loader:
                inputs = batch_inputs.to(device)
                noise = torch.randn_like(inputs) * 0.05
                noisy_inputs = torch.clamp(inputs + noise, 0.0, 1.0)

                optimizer.zero_grad()
                reconstructed = model(noisy_inputs)
                loss = criterion(reconstructed, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(loader)
        else:
            num_batches = 10
            for _ in range(num_batches):
                inputs = generate_synthetic_terrain_batch(
                    batch_size=batch_size, height=h, width=w, in_channels=in_channels
                ).to(device)
                noise = torch.randn_like(inputs) * 0.05
                noisy_inputs = torch.clamp(inputs + noise, 0.0, 1.0)

                optimizer.zero_grad()
                reconstructed = model(noisy_inputs)
                loss = criterion(reconstructed, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / num_batches

        if epoch % 3 == 0 or epoch == epochs:
            print(f"    Epoch [{epoch}/{epochs}] - Loss (MSE): {avg_loss:.6f}")

    print("[+] Autoencoder Training Complete!")

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Save PyTorch model checkpoint
    pt_path = os.path.join(output_dir, "autoencoder.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"[+] Saved PyTorch checkpoint: {pt_path}")

    # Central exports directory
    central_exports = os.path.abspath(os.path.join(output_dir, "../../exports"))
    os.makedirs(central_exports, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(central_exports, "autoencoder.pt"))

    # --- ONNX Export ---
    export_model = ConvAutoencoder(in_channels=in_channels).to(device)
    export_model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
    export_model.eval()
    export_model.requires_grad_(False)

    onnx_path = os.path.join(output_dir, "autoencoder.onnx")
    central_onnx_path = os.path.join(central_exports, "autoencoder.onnx")

    dummy_input = torch.randn(1, in_channels, h, w, device=device)

    print(f"[*] Exporting Autoencoder to ONNX format at: {onnx_path}")
    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_frame'],
        output_names=['reconstructed_baseline'],
        dynamic_axes={
            'input_frame': {0: 'batch_size', 2: 'height', 3: 'width'},
            'reconstructed_baseline': {0: 'batch_size', 2: 'height', 3: 'width'}
        },
        dynamo=False
    )
    import shutil
    shutil.copyfile(onnx_path, central_onnx_path)
    print(f"[+] Successfully exported ONNX model! File size: {os.path.getsize(onnx_path) / 1024:.2f} KB")
    print("[*] Model ready for INT8/FP16 TensorRT quantization on edge hardware.")
    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Convolutional Autoencoder (Stage 2)")
    parser.add_argument("--data-dir", type=str, default="../../data/patches",
                        help="Path to clean patch images. If omitted or not found, falls back to synthetic terrain.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    cli_args = parser.parse_args()

    train_and_export(
        data_dir=os.path.abspath(cli_args.data_dir) if cli_args.data_dir else None,
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        lr=cli_args.lr,
    )
