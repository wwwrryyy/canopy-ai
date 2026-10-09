"""
export.py
=========
PyTorch → ONNX export for the Convolutional Despeckler U-Net.

Exports the DespecklerUNet as a clean single-pass graph ready for high-throughput
TensorRT engine compilation on NVIDIA Jetson edge platforms.
"""

import os
import argparse
import torch

from model import DespecklerUNet


def export_onnx(args):
    device = torch.device("cpu")

    model = DespecklerUNet(
        in_channels=1,
        base_ch=args.base_ch,
    ).to(device)

    # Load trained weights if provided
    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded checkpoint: {args.checkpoint}")

    model.eval()

    dummy = torch.randn(1, 1, args.height, args.width, device=device)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("[*] Exporting Despeckler U-Net to ONNX format...")
    torch.onnx.export(
        model,
        dummy,
        args.output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["corrupted_frame"],
        output_names=["clean_frame"],
        dynamic_axes={
            "corrupted_frame": {0: "batch_size"},
            "clean_frame": {0: "batch_size"},
        },
        dynamo=False,
    )

    size_kb = os.path.getsize(args.output) / 1024
    print(f"[+] Exported: {args.output} ({size_kb:.1f} KB)")


def parse_args():
    p = argparse.ArgumentParser(description="Export Despeckler U-Net to ONNX")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to trained .pt checkpoint.")
    p.add_argument("--output", type=str, default="../../exports/despeckler.onnx",
                    help="Output ONNX file path.")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    return p.parse_args()


if __name__ == "__main__":
    export_onnx(parse_args())
