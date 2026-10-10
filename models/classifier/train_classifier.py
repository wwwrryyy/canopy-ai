"""
train_classifier.py
===================
Training and calibration pipeline for Stage 3 AgriculturalClassifier (MobileNetV3-Small).
Features:
- Transfer learning from ImageNet pre-trained backbone
- Cosine annealing learning rate scheduling and label smoothing
- Post-hoc temperature scaling to minimize Expected Calibration Error (ECE)
- Checkpoint persistence and optional automatic ONNX export
"""

import os
import sys
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Ensure local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from classifier import AgriculturalClassifier, CLASS_NAMES
from dataset import AgriculturalPathologyDataset


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    Measures the alignment between confidence scores and empirical accuracy.
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def calibrate_temperature(model: AgriculturalClassifier, val_loader: DataLoader, device: torch.device) -> float:
    """
    Fits post-hoc temperature scaling on validation set using Negative Log-Likelihood (NLL).
    """
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            # Fetch unscaled raw logits directly from net
            raw_logits = model.net(x)
            logits_list.append(raw_logits.cpu())
            labels_list.append(y.cpu())

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    # Initial ECE before calibration (T=1.0)
    initial_probs = torch.softmax(logits, dim=1)
    initial_ece = compute_ece(initial_probs, labels)

    # Optimize log-temperature to prevent negative or zero temperature
    log_temp = nn.Parameter(torch.zeros(1, device=device))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([log_temp], lr=0.01, max_iter=50)

    def eval_loss():
        optimizer.zero_grad()
        current_temp = torch.exp(log_temp)
        loss = criterion(logits.to(device) / current_temp, labels.to(device))
        loss.backward()
        return loss

    optimizer.step(eval_loss)

    opt_temp = float(torch.clamp(torch.exp(log_temp.data), min=0.05, max=10.0).item())
    model.temperature.data.fill_(opt_temp)

    calibrated_probs = torch.softmax(logits / opt_temp, dim=1)
    calibrated_ece = compute_ece(calibrated_probs, labels)

    print(f"[*] Calibration Complete: Temperature T = {opt_temp:.4f}")
    print(f"    - ECE before: {initial_ece:.4f}")
    print(f"    - ECE after:  {calibrated_ece:.4f}")

    return opt_temp


def train_classifier(args):
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"[*] Training AgriculturalClassifier on device: {device}")

    # 1. Dataset & DataLoaders with deterministic validation split (no augmentations)
    train_samples = int(args.samples_per_class * 0.8)
    val_samples = max(int(args.samples_per_class * 0.2), 10)

    train_ds = AgriculturalPathologyDataset(
        samples_per_class=train_samples,
        is_train=True,
        seed=42,
    )
    val_ds = AgriculturalPathologyDataset(
        samples_per_class=val_samples,
        is_train=False,
        seed=1337,
    )

    should_drop_last = len(train_ds) > args.batch_size
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=should_drop_last)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"[*] Dataset: {len(train_ds)} train samples, {len(val_ds)} val samples across {len(CLASS_NAMES)} classes")

    # 2. Model Initialization
    model = AgriculturalClassifier(
        in_channels=args.in_channels,
        num_classes=len(CLASS_NAMES),
        pretrained=args.pretrained,
    ).to(device)

    # 3. Optimizer & Criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_state = None

    # 4. Training Loop
    print(f"[*] Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            # Raw logits during training
            out = model.net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        scheduler.step()
        train_acc = correct / max(total, 1)
        train_loss = running_loss / max(total, 1)

        # Validation evaluation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model.net(x)
                preds = torch.argmax(out, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)

        val_acc = val_correct / max(val_total, 1)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Load best weights before calibration
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[+] Best Validation Accuracy: {best_val_acc * 100:.2f}%")

    # 5. Temperature Scaling Post-hoc Calibration
    print("[*] Calibrating softmax probabilities via temperature scaling...")
    calibrate_temperature(model, val_loader, device)

    # 6. Save Checkpoint
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"[+] Saved calibrated checkpoint to: {args.output}")

    # 7. Optional ONNX Export
    if args.export_onnx:
        from export import export_onnx
        export_args = argparse.Namespace(
            checkpoint=args.output,
            in_channels=args.in_channels,
            size=64,
            output=os.path.join(os.path.dirname(args.output), "..", "..", "exports", "classifier.onnx"),
        )
        export_onnx(export_args)


def parse_args():
    p = argparse.ArgumentParser(description="Train Stage 3 MobileNetV3-Small Agricultural Classifier")
    p.add_argument("--epochs", type=int, default=8, help="Number of training epochs (default: 8)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    p.add_argument("--samples-per-class", type=int, default=200, help="Synthesized samples per class (default: 200)")
    p.add_argument("--in-channels", type=int, default=3, help="Input channels (default: 3 for RGB)")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="Use ImageNet pretrained weights")
    p.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    p.add_argument("--output", type=str, default=os.path.join(SCRIPT_DIR, "classifier_best.pt"), help="Checkpoint output path")
    p.add_argument("--export-onnx", action=argparse.BooleanOptionalAction, default=True, help="Export to exports/classifier.onnx upon completion")
    return p.parse_args()


if __name__ == "__main__":
    train_classifier(parse_args())
