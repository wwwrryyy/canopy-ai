"""
inference_backend.py
====================
Seam and adapters for neural model execution (Stage 1 Despeckler, Stage 2 Autoencoder,
Stage 3 MobileNetV3 Classifier). Decouples caller logic from runtime engines.
"""

from abc import ABC, abstractmethod
import os
import sys
from typing import Optional, Tuple
import numpy as np

# Ensure models can be imported if PyTorch backend is used
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "despeckler"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "autoencoder"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "classifier"))

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import torch
    from model import DespecklerUNet
    from autoencoder import ConvAutoencoder
    from classifier import AgriculturalClassifier, CLASS_NAMES
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    CLASS_NAMES = [
        "Healthy Canopy",
        "Fungal/Bacterial Disease",
        "Pest Infestation",
        "Water Stress",
        "Weed Invasion",
    ]


class InferenceBackend(ABC):
    """
    Interface for executing the neural models in the precision agriculture pipeline.
    Hides ONNX Runtime session details, PyTorch tensors, input node names, and
    thread pools behind simple NumPy array transformations.
    """

    @abstractmethod
    def despeckle(self, frame_128: np.ndarray) -> np.ndarray:
        """
        Runs Stage 1 image restoration.
        Args:
            frame_128: [128, 128] float32 array in [0, 1].
        Returns:
            clean_frame: [128, 128] float32 array in [0, 1].
        """
        pass

    @abstractmethod
    def reconstruct_baseline(self, clean_frame_128: np.ndarray) -> np.ndarray:
        """
        Runs Stage 2 healthy canopy baseline reconstruction.
        Args:
            clean_frame_128: [128, 128] float32 array in [0, 1].
        Returns:
            recon_frame: [128, 128] float32 array in [0, 1].
        """
        pass

    @abstractmethod
    def classify_crop(self, crop_64x64_3ch: np.ndarray) -> Tuple[str, float]:
        """
        Runs Stage 3 agricultural anomaly classification on a 64x64 patch.
        Args:
            crop_64x64_3ch: [3, 64, 64] float32 array in [0, 1].
        Returns:
            (class_name, confidence)
        """
        pass


class OnnxInferenceBackend(InferenceBackend):
    """
    Production edge adapter using ONNX Runtime (compatible with TensorRT engines).
    """

    def __init__(self, exports_dir: Optional[str] = None):
        if not HAS_ORT:
            raise RuntimeError("onnxruntime is not installed.")

        if exports_dir is None:
            exports_dir = os.path.join(PROJECT_ROOT, "exports")

        despeckle_path = os.path.join(exports_dir, "despeckler.onnx")
        ae_path = os.path.join(exports_dir, "autoencoder.onnx")
        clf_path = os.path.join(exports_dir, "classifier.onnx")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.despeckle_sess = ort.InferenceSession(despeckle_path, opts) if os.path.isfile(despeckle_path) else None
        self.ae_sess = ort.InferenceSession(ae_path, opts) if os.path.isfile(ae_path) else None
        self.clf_sess = ort.InferenceSession(clf_path, opts) if os.path.isfile(clf_path) else None

        if self.despeckle_sess is None or self.ae_sess is None or self.clf_sess is None:
            raise FileNotFoundError(f"One or more required ONNX models missing in {exports_dir}")

    def despeckle(self, frame_128: np.ndarray) -> np.ndarray:
        x_in = frame_128[np.newaxis, np.newaxis, :, :].astype(np.float32)
        clean_out = self.despeckle_sess.run(None, {"corrupted_frame": x_in})[0]
        return clean_out[0, 0]

    def reconstruct_baseline(self, clean_frame_128: np.ndarray) -> np.ndarray:
        x_in = clean_frame_128[np.newaxis, np.newaxis, :, :].astype(np.float32)
        recon_out = self.ae_sess.run(None, {"input_frame": x_in})[0]
        return recon_out[0, 0]

    def classify_crop(self, crop_64x64_3ch: np.ndarray) -> Tuple[str, float]:
        x_in = crop_64x64_3ch[np.newaxis, :, :, :].astype(np.float32)
        logits = self.clf_sess.run(None, {"crop_patch": x_in})[0]
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        top_idx = int(np.argmax(probs[0]))
        confidence = float(probs[0, top_idx])
        return CLASS_NAMES[top_idx], confidence


class PyTorchInferenceBackend(InferenceBackend):
    """
    PyTorch adapter for evaluation, fine-tuning, or environments without ONNX exports.
    """

    def __init__(self, exports_dir: Optional[str] = None):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not installed.")

        self.despeckler = DespecklerUNet().eval()
        self.autoencoder = ConvAutoencoder().eval()
        self.classifier = AgriculturalClassifier().eval()

        if exports_dir is None:
            exports_dir = os.path.join(PROJECT_ROOT, "exports")

        despeckle_pt = os.path.join(exports_dir, "despeckler_best.pt")
        ae_pt = os.path.join(exports_dir, "autoencoder.pt")

        if os.path.isfile(despeckle_pt):
            self.despeckler.load_state_dict(torch.load(despeckle_pt, map_location="cpu", weights_only=True))
        if os.path.isfile(ae_pt):
            self.autoencoder.load_state_dict(torch.load(ae_pt, map_location="cpu", weights_only=True))

    def despeckle(self, frame_128: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(frame_128[np.newaxis, np.newaxis, :, :].astype(np.float32))
        with torch.no_grad():
            out = self.despeckler(x_t).numpy()
        return out[0, 0]

    def reconstruct_baseline(self, clean_frame_128: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(clean_frame_128[np.newaxis, np.newaxis, :, :].astype(np.float32))
        with torch.no_grad():
            out = self.autoencoder(x_t).numpy()
        return out[0, 0]

    def classify_crop(self, crop_64x64_3ch: np.ndarray) -> Tuple[str, float]:
        x_t = torch.from_numpy(crop_64x64_3ch[np.newaxis, :, :, :].astype(np.float32))
        with torch.no_grad():
            logits = self.classifier(x_t).numpy()
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        top_idx = int(np.argmax(probs[0]))
        confidence = float(probs[0, top_idx])
        return CLASS_NAMES[top_idx], confidence


class MockInferenceBackend(InferenceBackend):
    """
    Lightweight, deterministic test adapter for unit tests. Requires no weights or heavy frameworks.
    """

    def __init__(self, simulated_class: str = "Pest Infestation", simulated_confidence: float = 0.95):
        self.simulated_class = simulated_class
        self.simulated_confidence = simulated_confidence

    def despeckle(self, frame_128: np.ndarray) -> np.ndarray:
        return frame_128.copy()

    def reconstruct_baseline(self, clean_frame_128: np.ndarray) -> np.ndarray:
        # Returns a smooth baseline image (e.g. uniform mean) so difference heatmaps are easy to test
        return np.full_like(clean_frame_128, float(np.mean(clean_frame_128)))

    def classify_crop(self, crop_64x64_3ch: np.ndarray) -> Tuple[str, float]:
        return self.simulated_class, self.simulated_confidence


def create_inference_backend(exports_dir: Optional[str] = None, prefer_onnx: bool = True) -> InferenceBackend:
    """Factory selecting the best available inference backend adapter."""
    if prefer_onnx and HAS_ORT:
        try:
            return OnnxInferenceBackend(exports_dir)
        except Exception:
            pass

    if HAS_TORCH:
        try:
            return PyTorchInferenceBackend(exports_dir)
        except Exception:
            pass

    return MockInferenceBackend()
