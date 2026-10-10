"""
export.py
=========
PyTorch → ONNX export for Stage 3 AgriculturalClassifier (MobileNetV3-Small).
"""

import os
import argparse
import torch

from classifier import AgriculturalClassifier, CLASS_NAMES


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def export_onnx(args):
    device = torch.device("cpu")

    model = AgriculturalClassifier(
        in_channels=args.in_channels,
        num_classes=len(CLASS_NAMES),
    ).to(device)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded checkpoint: {args.checkpoint} (temperature: {float(model.temperature.data.item()):.4f})")

    model.eval()

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dummy = torch.randn(1, args.in_channels, args.size, args.size, device=device)

    print("[*] Exporting AgriculturalClassifier to ONNX format...")
    torch.onnx.export(
        model,
        dummy,
        output_path,
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

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[+] Exported: {output_path} ({size_kb:.1f} KB)")

    # Parity verification with onnxruntime if available
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(output_path)
        ort_out = sess.run(["class_logits"], {"crop_patch": dummy.numpy()})[0]
        with torch.no_grad():
            pt_out = model(dummy).numpy()

        max_err = float(np.max(np.abs(ort_out - pt_out)))
        print(f"[+] ONNX vs PyTorch max absolute error: {max_err:.2e}")
        assert max_err < 1e-4, f"Parity mismatch: max error {max_err}"
    except ImportError:
        pass


def parse_args():
    default_export = os.path.join(PROJECT_ROOT, "exports", "classifier.onnx")
    p = argparse.ArgumentParser(description="Export AgriculturalClassifier to ONNX")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to trained .pt checkpoint.")
    p.add_argument("--in-channels", type=int, default=3,
                   help="Input channels (3 for RGB, 7 for multispectral).")
    p.add_argument("--size", type=int, default=64,
                   help="Patch height/width (default 64).")
    p.add_argument("--output", type=str, default=default_export,
                   help="Output path for ONNX file.")
    return p.parse_args()


if __name__ == "__main__":
    export_onnx(parse_args())
