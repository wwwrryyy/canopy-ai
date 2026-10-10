"""
test_classifier.py
==================
Unit and integration tests for Stage 3 AgriculturalClassifier:
- Dataset generation and tensor shapes
- Model architecture and temperature scaling calibration
- PyTorch vs ONNX Runtime parity (< 1e-4)
- Confidence gating in CanopyInspectionModule
- Real inference execution through OnnxInferenceBackend
"""

import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEPLOYMENT_DIR = os.path.join(PROJECT_ROOT, "deployment")
CLASSIFIER_DIR = os.path.join(PROJECT_ROOT, "models", "classifier")

sys.path.insert(0, DEPLOYMENT_DIR)
sys.path.insert(0, CLASSIFIER_DIR)

import cv2
from canopy_anomaly import CanopyAnomaly
from canopy_inspection import CanopyInspectionModule
from inference_backend import MockInferenceBackend, OnnxInferenceBackend, HAS_ORT

try:
    import torch
    from classifier import AgriculturalClassifier, CLASS_NAMES
    from dataset import AgriculturalPathologyDataset
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


class TestAgriculturalClassifier(unittest.TestCase):

    def test_dataset_shapes_and_class_balance(self):
        """Verifies synthetic pathology dataset produces balanced [3, 64, 64] float tensors in [0, 1]."""
        if not HAS_TORCH:
            self.skipTest("torch not installed")
        ds = AgriculturalPathologyDataset(samples_per_class=5, is_train=False)
        self.assertEqual(len(ds), 25)

        for i in range(len(ds)):
            tensor, label = ds[i]
            self.assertEqual(tensor.shape, (3, 64, 64))
            self.assertIn(label, list(range(len(CLASS_NAMES))))
            self.assertGreaterEqual(float(tensor.min()), 0.0)
            self.assertLessEqual(float(tensor.max()), 1.0)

    def test_model_forward_and_probabilities(self):
        """Verifies AgriculturalClassifier forward pass and temperature scaling probability normalization."""
        try:
            import torch
            model = AgriculturalClassifier(in_channels=3, num_classes=5)
            model.eval()

            x = torch.randn(2, 3, 64, 64)
            logits = model(x)
            self.assertEqual(logits.shape, (2, 5))

            probs = model.predict_proba(x)
            self.assertEqual(probs.shape, (2, 5))

            sums = probs.sum(dim=1).numpy()
            np.testing.assert_allclose(sums, np.ones(2), rtol=1e-5)
        except ImportError:
            pass

    def test_onnx_export_parity(self):
        """Verifies exports/classifier.onnx achieves numerical parity (< 1e-4) with PyTorch checkpoint."""
        onnx_path = os.path.join(PROJECT_ROOT, "exports", "classifier.onnx")
        checkpoint_path = os.path.join(PROJECT_ROOT, "models", "classifier", "classifier_best.pt")

        if not os.path.isfile(onnx_path) or not HAS_ORT:
            self.skipTest("classifier.onnx or onnxruntime not available")

        try:
            import torch
            import onnxruntime as ort

            model = AgriculturalClassifier(in_channels=3, num_classes=5)
            if os.path.isfile(checkpoint_path):
                state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
            model.eval()

            dummy = torch.randn(1, 3, 64, 64)

            # PyTorch output
            with torch.no_grad():
                pt_logits = model(dummy).numpy()

            # ONNX Runtime output
            sess = ort.InferenceSession(onnx_path)
            ort_logits = sess.run(["class_logits"], {"crop_patch": dummy.numpy()})[0]

            max_err = float(np.max(np.abs(pt_logits - ort_logits)))
            self.assertLess(max_err, 1e-4, f"Parity mismatch between PyTorch and ONNX: {max_err}")
        except ImportError:
            pass

    def test_confidence_gating_unclassified_fallback(self):
        """Low-confidence predictions below min_confidence must be gated as 'Unclassified Crop Stress'."""
        backend = MockInferenceBackend(
            simulated_class="Fungal/Bacterial Disease",
            simulated_confidence=0.35,  # Below default min_confidence=0.50
        )
        module = CanopyInspectionModule(
            inference_backend=backend,
            mse_threshold=0.01,
            min_confidence=0.50,
        )

        frame = np.full((128, 128), 0.5, dtype=np.float32)
        cv2.circle(frame, (64, 64), 10, 1.0, -1)

        anomalies = module.inspect(frame)
        self.assertGreaterEqual(len(anomalies), 1)
        anom = anomalies[0]
        self.assertEqual(anom.anomaly_type, "Unclassified Crop Stress")
        self.assertAlmostEqual(anom.confidence, 0.35, places=2)

    def test_confidence_gating_high_confidence_passthrough(self):
        """High-confidence predictions >= min_confidence retain their classified pathology."""
        backend = MockInferenceBackend(
            simulated_class="Fungal/Bacterial Disease",
            simulated_confidence=0.88,  # Above min_confidence=0.50
        )
        module = CanopyInspectionModule(
            inference_backend=backend,
            mse_threshold=0.01,
            min_confidence=0.50,
        )

        frame = np.full((128, 128), 0.5, dtype=np.float32)
        cv2.circle(frame, (64, 64), 10, 1.0, -1)

        anomalies = module.inspect(frame)
        self.assertGreaterEqual(len(anomalies), 1)
        anom = anomalies[0]
        self.assertEqual(anom.anomaly_type, "Fungal/Bacterial Disease")
        self.assertAlmostEqual(anom.confidence, 0.88, places=2)

    def test_onnx_backend_real_inference_execution(self):
        """Verifies real OnnxInferenceBackend executes classify_crop and produces valid class and confidence."""
        onnx_path = os.path.join(PROJECT_ROOT, "exports", "classifier.onnx")
        if not os.path.isfile(onnx_path) or not HAS_ORT:
            self.skipTest("classifier.onnx or onnxruntime not available")

        backend = OnnxInferenceBackend()
        crop_patch = np.random.uniform(0.2, 0.8, (3, 64, 64)).astype(np.float32)

        class_name, confidence = backend.classify_crop(crop_patch)
        self.assertIn(class_name, CLASS_NAMES)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_external_dataset_loading_and_split(self):
        """Verifies AgriculturalPathologyDataset loads from external_data_dir and partitions train/val."""
        if not HAS_TORCH:
            self.skipTest("torch not installed")
        external_dir = os.path.join(PROJECT_ROOT, "data", "external_pathologies")
        if not os.path.isdir(external_dir):
            self.skipTest("data/external_pathologies not yet created")

        train_ds = AgriculturalPathologyDataset(external_data_dir=external_dir, is_train=True, val_split=0.2, seed=42)
        val_ds = AgriculturalPathologyDataset(external_data_dir=external_dir, is_train=False, val_split=0.2, seed=42)

        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(val_ds), 0)
        self.assertGreater(len(train_ds.external_images), 0)
        self.assertGreater(len(val_ds.external_images), 0)
        # Check no intersection between train and val image paths
        train_paths = {p for p, _ in train_ds.external_images}
        val_paths = {p for p, _ in val_ds.external_images}
        self.assertEqual(len(train_paths.intersection(val_paths)), 0)

        tensor, label = train_ds[0]
        self.assertEqual(tensor.shape, (3, 64, 64))
        self.assertIn(label, list(range(len(CLASS_NAMES))))

    def test_external_dataset_nonexistent_dir_raises(self):
        """Verifies FileNotFoundError is raised when external_data_dir does not exist."""
        if not HAS_TORCH:
            self.skipTest("torch not installed")
        nonexistent = os.path.join(PROJECT_ROOT, "data", "nonexistent_dir_xyz")
        with self.assertRaises(FileNotFoundError):
            AgriculturalPathologyDataset(external_data_dir=nonexistent)

    def test_external_dataset_missing_class_raises(self):
        """Verifies ValueError is raised when any required CLASS_NAMES folder is missing."""
        if not HAS_TORCH:
            self.skipTest("torch not installed")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create only one class folder out of five
            single_class_dir = os.path.join(tmp_dir, "Healthy Canopy")
            os.makedirs(single_class_dir, exist_ok=True)
            dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(os.path.join(single_class_dir, "img1.png"), dummy_img))
            self.assertTrue(cv2.imwrite(os.path.join(single_class_dir, "img2.png"), dummy_img))
            with self.assertRaises(ValueError) as ctx:
                AgriculturalPathologyDataset(external_data_dir=tmp_dir)
            self.assertIn("missing required class folder", str(ctx.exception))

    def test_external_dataset_single_image_class_raises(self):
        """Verifies ValueError is raised when an external class has fewer than 2 images to split."""
        if not HAS_TORCH:
            self.skipTest("torch not installed")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
            # Create all 5 classes, but leave one with only 1 image
            for idx, cname in enumerate(CLASS_NAMES):
                cdir = os.path.join(tmp_dir, cname.replace("/", "_"))
                os.makedirs(cdir, exist_ok=True)
                count = 1 if idx == 0 else 2
                for i in range(count):
                    self.assertTrue(cv2.imwrite(os.path.join(cdir, f"img_{i}.png"), dummy_img))

            with self.assertRaises(ValueError) as ctx:
                AgriculturalPathologyDataset(external_data_dir=tmp_dir)
            self.assertIn("At least 2 images are required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
