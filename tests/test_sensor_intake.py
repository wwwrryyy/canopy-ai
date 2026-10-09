"""
test_sensor_intake.py
=====================
Unit tests verifying the SensorIntakeModule and MultispectralCanopyFrame domain model.
"""

import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEPLOYMENT_DIR = os.path.join(PROJECT_ROOT, "deployment")
sys.path.insert(0, DEPLOYMENT_DIR)

from sensor_intake import (
    SensorIntakeModule,
    MultispectralCanopyFrame,
    compute_ndvi,
    compute_ndre,
    normalize_index,
    append_vegetation_indices,
)
from inference_backend import MockInferenceBackend
from difference_engine import OpenCVDifferenceAdapter
from canopy_inspection import CanopyInspectionModule


class TestSensorIntakeModule(unittest.TestCase):

    def setUp(self):
        self.intake = SensorIntakeModule(target_size=(128, 128))

    def test_radiometric_indices_math(self):
        """Verifies NDVI and NDRE mathematical formulas."""
        nir = np.full((10, 10), 0.8, dtype=np.float32)
        red = np.full((10, 10), 0.2, dtype=np.float32)

        # NDVI = (0.8 - 0.2) / (0.8 + 0.2) = 0.6 / 1.0 = 0.6
        raw_ndvi = compute_ndvi(nir, red)
        self.assertAlmostEqual(float(raw_ndvi[0, 0]), 0.6, places=4)

        # normalize_index maps 0.6 -> (0.6 + 1.0) / 2.0 = 0.8
        norm_ndvi = normalize_index(raw_ndvi)
        self.assertAlmostEqual(float(norm_ndvi[0, 0]), 0.8, places=4)

    def test_ingest_5band_multispectral_synthesizes_ndvi_and_ndre(self):
        """Ingests 5-band payload [B, G, R, RedEdge, NIR] and verifies dynamic index calculation."""
        # Shape: [200, 200, 5]
        arr_5b = np.zeros((200, 200, 5), dtype=np.float32)
        arr_5b[:, :, 0] = 0.1  # Blue
        arr_5b[:, :, 1] = 0.4  # Green
        arr_5b[:, :, 2] = 0.2  # Red
        arr_5b[:, :, 3] = 0.3  # RedEdge
        arr_5b[:, :, 4] = 0.8  # NIR

        frame = self.intake.ingest(arr_5b)
        self.assertIsInstance(frame, MultispectralCanopyFrame)
        self.assertEqual(frame.height, 128)
        self.assertEqual(frame.width, 128)
        self.assertTrue(frame.has_vegetation_indices)
        self.assertIsNotNone(frame.ndvi)
        self.assertIsNotNone(frame.ndre)

        # Verified normalized NDVI: (0.8-0.2)/(0.8+0.2)=0.6 -> normalized 0.8
        self.assertAlmostEqual(float(frame.ndvi[0, 0]), 0.8, places=3)

        # Test projections
        single_ch = frame.to_single_channel(prefer_ndvi=True)
        self.assertEqual(single_ch.shape, (128, 128))
        self.assertAlmostEqual(float(single_ch[0, 0]), 0.8, places=3)

        rgb_comp = frame.to_rgb_composite()
        self.assertEqual(rgb_comp.shape, (3, 128, 128))
        self.assertAlmostEqual(float(rgb_comp[0, 0, 0]), 0.2, places=3)  # Red channel

        tensor_7ch = frame.to_multispectral_tensor()
        self.assertEqual(tensor_7ch.shape, (7, 128, 128))

    def test_ingest_3band_rgb_payload(self):
        """Ingests 3-band RGB image and provides neutral vegetation stubs."""
        arr_rgb = np.full((128, 128, 3), 128, dtype=np.uint8)  # uint8 automatic conversion
        frame = self.intake.ingest(arr_rgb)

        self.assertEqual(frame.height, 128)
        self.assertEqual(frame.width, 128)
        self.assertIn("red", frame.bands)
        self.assertIn("green", frame.bands)
        self.assertIn("blue", frame.bands)
        # Neutral stubs should be 0.5
        self.assertAlmostEqual(float(frame.ndvi[0, 0]), 0.5, places=3)

        rgb_comp = frame.to_rgb_composite()
        self.assertEqual(rgb_comp.shape, (3, 128, 128))

    def test_ingest_dict_of_band_arrays(self):
        """Ingests dictionary of named bands."""
        band_dict = {
            "blue": np.full((100, 100), 0.1, dtype=np.float32),
            "green": np.full((100, 100), 0.5, dtype=np.float32),
            "red": np.full((100, 100), 0.2, dtype=np.float32),
            "red_edge": np.full((100, 100), 0.4, dtype=np.float32),
            "nir": np.full((100, 100), 0.9, dtype=np.float32),
        }
        frame = self.intake.ingest(band_dict)
        self.assertEqual(frame.height, 128)
        self.assertTrue(frame.has_vegetation_indices)
        self.assertGreater(float(frame.ndvi[0, 0]), 0.7)

    def test_canopy_inspection_accepts_raw_5band_multispectral(self):
        """End-to-end integration: CanopyInspectionModule.inspect directly accepts 5-band arrays."""
        backend = MockInferenceBackend(simulated_class="Water Stress", simulated_confidence=0.94)
        diff_engine = OpenCVDifferenceAdapter()
        pipeline = CanopyInspectionModule(
            inference_backend=backend,
            difference_engine=diff_engine,
            sensor_intake=self.intake,
            mse_threshold=0.01,
        )

        # 5-band input with anomaly spot in Red/NIR
        arr_5b = np.full((128, 128, 5), 0.5, dtype=np.float32)
        # Inject anomalous dry/stressed spot (low NIR, high Red)
        arr_5b[50:70, 50:70, 2] = 0.9  # high red
        arr_5b[50:70, 50:70, 4] = 0.1  # low nir

        diag = pipeline.inspect_with_diagnostics(arr_5b)
        self.assertIsInstance(diag.canopy_frame, MultispectralCanopyFrame)
        self.assertIn("stage0_sensor_intake", diag.latencies_ms)
        self.assertGreaterEqual(len(diag.anomalies), 1)
        self.assertEqual(diag.anomalies[0].anomaly_type, "Water Stress")

    def test_append_vegetation_indices_backward_compatibility(self):
        """Verifies append_vegetation_indices matches historical shape and scale."""
        bands_5b = np.full((64, 64, 5), 0.5, dtype=np.float32)
        out_7b = append_vegetation_indices(bands_5b)
        self.assertEqual(out_7b.shape, (64, 64, 7))


if __name__ == "__main__":
    unittest.main()
