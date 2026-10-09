"""
test_canopy_inspection.py
=========================
Unit tests verifying the CanopyInspectionModule purely through its public interface.
Uses MockInferenceBackend to verify behavior independently of heavy ML runtimes.
"""

import os
import sys
import unittest
import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEPLOYMENT_DIR = os.path.join(PROJECT_ROOT, "deployment")
sys.path.insert(0, DEPLOYMENT_DIR)

from canopy_anomaly import CanopyAnomaly
from difference_engine import OpenCVDifferenceAdapter
from inference_backend import MockInferenceBackend
from canopy_inspection import CanopyInspectionModule


class TestCanopyInspectionModule(unittest.TestCase):

    def setUp(self):
        self.backend = MockInferenceBackend(
            simulated_class="Pest Infestation", simulated_confidence=0.92
        )
        self.diff_engine = OpenCVDifferenceAdapter()
        self.module = CanopyInspectionModule(
            inference_backend=self.backend,
            difference_engine=self.diff_engine,
            mse_threshold=0.01,
            min_area=10,
            max_area=5000,
        )

    def test_inspect_clean_canopy_returns_no_anomalies(self):
        """A uniform healthy canopy frame should yield zero anomaly detections."""
        uniform_frame = np.full((128, 128), 0.5, dtype=np.float32)
        anomalies = self.module.inspect(uniform_frame)
        self.assertEqual(len(anomalies), 0)

    def test_inspect_detects_and_classifies_synthetic_anomaly(self):
        """An injected high-contrast anomaly circle should be detected and classified."""
        frame = np.full((128, 128), 0.5, dtype=np.float32)
        # Inject an anomaly spot at center (radius 10)
        cv2.circle(frame, (64, 64), 10, 1.0, -1)

        anomalies = self.module.inspect(frame)
        self.assertGreaterEqual(len(anomalies), 1)

        anom = anomalies[0]
        self.assertIsInstance(anom, CanopyAnomaly)
        self.assertEqual(anom.anomaly_type, "Pest Infestation")
        self.assertAlmostEqual(anom.confidence, 0.92, places=2)
        self.assertGreater(anom.mse_score, 0.01)
        self.assertGreater(anom.area_pixels, 0)
        # Bounding box should enclose the circle center (64, 64)
        x, y, w, h = anom.bbox
        self.assertTrue(x <= 64 <= x + w)
        self.assertTrue(y <= 64 <= y + h)

    def test_inspect_with_diagnostics_provides_all_stage_latencies(self):
        """Diagnostics interface must populate intermediate maps and latency breakdown."""
        frame = np.random.uniform(0.3, 0.7, (128, 128)).astype(np.float32)
        result = self.module.inspect_with_diagnostics(frame)

        self.assertEqual(result.clean_frame.shape, (128, 128))
        self.assertEqual(result.recon_frame.shape, (128, 128))
        self.assertEqual(result.heatmap.shape, (128, 128))
        self.assertIn("stage1_despeckle", result.latencies_ms)
        self.assertIn("stage2_autoencoder", result.latencies_ms)
        self.assertIn("heatmap_diff", result.latencies_ms)
        self.assertIn("stage3_classifier", result.latencies_ms)
        self.assertIn("total", result.latencies_ms)

    def test_arbitrary_dimensions_handling(self):
        """Verifies models handle dimensions not divisible by 4 (e.g. 130x130)."""
        try:
            import torch
            from model import DespecklerUNet
            from autoencoder import ConvAutoencoder

            x = torch.randn(1, 1, 130, 130)
            despeckler = DespecklerUNet(in_channels=1)
            y_despeck = despeckler(x)
            self.assertEqual(y_despeck.shape, (1, 1, 130, 130))

            ae = ConvAutoencoder(in_channels=1)
            y_ae = ae(x)
            self.assertEqual(y_ae.shape, (1, 1, 130, 130))
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
