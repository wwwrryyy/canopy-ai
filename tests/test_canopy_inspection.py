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
from difference_engine import DifferenceConfig, OpenCVDifferenceAdapter, NativeDifferenceAdapter
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

    def test_cloud_shadow_ramp_suppressed_with_cloud_shadow_resilient_preset(self):
        """A 50% lighting gradient across healthy canopy must not trigger false positive anomalies."""
        np.random.seed(42)
        x = np.linspace(0, 8 * np.pi, 128)
        rows = (np.sin(x)[:, None] * 0.15 + 0.6).astype(np.float32)
        noise = np.random.normal(0, 0.02, (128, 128)).astype(np.float32)
        base_frame = np.clip(rows + noise, 0.1, 0.9)

        shadow_ramp = np.linspace(1.0, 0.5, 128, dtype=np.float32)[:, None]
        shadowed_frame = base_frame * shadow_ramp

        backend = MockInferenceBackend(
            simulated_class="Pest Infestation",
            simulated_confidence=0.92,
            baseline_frame=base_frame,
        )
        module_resilient = CanopyInspectionModule(
            inference_backend=backend,
            difference_config=DifferenceConfig.cloud_shadow_resilient(),
            mse_threshold=0.20,
            min_area=10,
        )
        anomalies = module_resilient.inspect(shadowed_frame)
        self.assertEqual(len(anomalies), 0)

    def test_pathological_structural_lesion_detected_under_shadow(self):
        """Localized structural leaf necrosis / pest defoliation is detected even under shadow."""
        np.random.seed(42)
        x = np.linspace(0, 8 * np.pi, 128)
        rows = (np.sin(x)[:, None] * 0.15 + 0.6).astype(np.float32)
        noise = np.random.normal(0, 0.02, (128, 128)).astype(np.float32)
        base_frame = np.clip(rows + noise, 0.1, 0.9)

        shadow_ramp = np.linspace(1.0, 0.5, 128, dtype=np.float32)[:, None]
        shadowed_frame = base_frame * shadow_ramp
        # Inject structural necrotic lesion at (64, 64) with radius 12
        cv2.circle(shadowed_frame, (64, 64), 12, 0.05, -1)

        backend = MockInferenceBackend(
            simulated_class="Pest Infestation",
            simulated_confidence=0.92,
            baseline_frame=base_frame,
        )
        module_resilient = CanopyInspectionModule(
            inference_backend=backend,
            difference_config=DifferenceConfig.cloud_shadow_resilient(),
            mse_threshold=0.20,
            min_area=10,
        )
        anomalies = module_resilient.inspect(shadowed_frame)
        self.assertGreaterEqual(len(anomalies), 1)
        anom = anomalies[0]
        self.assertEqual(anom.anomaly_type, "Pest Infestation")
        x_bb, y_bb, w_bb, h_bb = anom.bbox
        self.assertTrue(x_bb <= 64 <= x_bb + w_bb)
        self.assertTrue(y_bb <= 64 <= y_bb + h_bb)

    def test_difference_engine_presets_instantiation(self):
        """Validates all presets configure weights as expected."""
        balanced = DifferenceConfig.balanced()
        self.assertEqual(balanced.w_dssim, 0.5)
        self.assertEqual(balanced.w_lncc, 0.3)
        self.assertEqual(balanced.w_mse, 0.2)

        shadow = DifferenceConfig.cloud_shadow_resilient()
        self.assertEqual(shadow.w_dssim, 0.6)
        self.assertEqual(shadow.w_lncc, 0.3)
        self.assertEqual(shadow.w_mse, 0.1)

        sens = DifferenceConfig.high_sensitivity()
        self.assertEqual(sens.w_dssim, 0.4)
        self.assertEqual(sens.w_lncc, 0.3)
        self.assertEqual(sens.w_mse, 0.3)

    def test_native_and_opencv_difference_parity(self):
        """Verifies native C++ and Python OpenCV difference engines achieve numerical parity."""
        native_adapter = NativeDifferenceAdapter()
        opencv_adapter = OpenCVDifferenceAdapter()

        clean = np.random.uniform(0.3, 0.7, (128, 128)).astype(np.float32)
        recon = clean.copy()
        cv2.circle(clean, (64, 64), 10, 1.0, -1)

        hm_native = native_adapter.compute_heatmap(clean, recon)
        hm_cv = opencv_adapter.compute_heatmap(clean, recon)

        max_err = float(np.max(np.abs(hm_native - hm_cv)))
        self.assertLess(max_err, 1e-4)

    def test_difference_config_invalid_window_size(self):
        """Window size must be a positive odd integer."""
        with self.assertRaises(ValueError):
            DifferenceConfig(window_size=10)
        with self.assertRaises(ValueError):
            DifferenceConfig(window_size=0)
        with self.assertRaises(ValueError):
            DifferenceConfig(window_size=-5)

    def test_native_and_opencv_difference_custom_constants_parity(self):
        """Verifies parity when custom c1, c2, and window_size are supplied."""
        custom_cfg = DifferenceConfig(window_size=15, sigma=2.0, c1=0.0005, c2=0.002)
        native_adapter = NativeDifferenceAdapter(config=custom_cfg)
        opencv_adapter = OpenCVDifferenceAdapter(config=custom_cfg)

        clean = np.random.uniform(0.3, 0.7, (128, 128)).astype(np.float32)
        recon = clean.copy()
        cv2.circle(clean, (40, 40), 8, 0.9, -1)

        hm_native = native_adapter.compute_heatmap(clean, recon)
        hm_cv = opencv_adapter.compute_heatmap(clean, recon)

        max_err = float(np.max(np.abs(hm_native - hm_cv)))
        self.assertLess(max_err, 1e-4)


if __name__ == "__main__":
    unittest.main()
