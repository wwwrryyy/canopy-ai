"""
export.py
=========
PyTorch → ONNX export for Stage 3 AgriculturalClassifier (MobileNetV3-Small).
"""

import os
import argparse
import torch

from classifier import AgriculturalClassifier, CLASS_NAMES


def export_onnx(args):
    device = torch.device("cpu")

    model = AgriculturalClassifier(
        in_channels=args.in_channels,
        num_classes=len(CLASS_NAMES),
    ).to(device)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded checkpoint: {args.checkpoint}")

    model.eval()

    dummy = torch.randn(1, args.in_channels, args.size, args.size, device=device)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("[*] Exporting AgriculturalClassifier to ONNX format...")
    torch.onnx.export(
        model,
        dummy,
        args.output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["crop_patch"],
        output_names=["class_logits"],
        dynamic_axes={
            "crop_patch": {0: "batch_size"},
            "class_logits": {0: "batch_size"},
        },
        dynamo=False,
    )

    size_kb = os.path.getsize(args.output) / 1024
    print(f"[+] Exported: {args.output} ({size_kb:.1f} KB)")


def parse_args():
    p = argparse.ArgumentParser(description="Export AgriculturalClassifier to ONNX")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to trained .pt checkpoint.")
    p.add_argument("--in-channels", type=int, default=3,
                   help="Input channels (3 for RGB, 7 for multispectral).")
    p.add_argument("--size", type=int, default=64,
                   help="Patch height/width (default 64).")
    p.add_argument("--output", type=str, default="classifier.onnx",
                   help="Output path for ONNX file.")
    return p.parse_args()


if __name__ == "__main__":
    export_onnx(parse_args())
