"""
test_flight_telemetry.py
========================
Unit tests for the FlightTelemetrySpooler module verifying batching, schema contract,
and backwards compatibility.
"""

import os
import sys
import unittest
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEPLOYMENT_DIR = os.path.join(PROJECT_ROOT, "deployment")
sys.path.insert(0, DEPLOYMENT_DIR)

from canopy_anomaly import CanopyAnomaly
from flight_telemetry import FlightTelemetrySpooler
from sharp_sync_daemon import SHARPOfflineResilienceManager


class TestFlightTelemetrySpooler(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_telemetry.db")
        self.telemetry = FlightTelemetrySpooler(
            db_path=self.db_path, batch_size=5, auto_start_monitor=False
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_record_canopy_anomaly_dataclass(self):
        """Records typed CanopyAnomaly instance and verifies persisted columns."""
        anom = CanopyAnomaly(
            anomaly_id=42,
            anomaly_type="Weed Invasion",
            confidence=0.88,
            mse_score=0.045,
            bbox=(10, 20, 30, 40),
            area_pixels=1200,
        )

        self.telemetry.record(anom, latitude=37.77, longitude=-122.41, altitude=50.0)
        self.telemetry.flush()

        logs = self.telemetry.get_recent_logs(limit=1)
        self.assertEqual(len(logs), 1)
        log = logs[0]
        self.assertEqual(log["anomaly_id"], 42)
        self.assertEqual(log["type"], "Weed Invasion")
        self.assertAlmostEqual(log["confidence"], 0.88, places=2)
        self.assertAlmostEqual(log["mse_score"], 0.045, places=3)
        self.assertEqual(log["bbox"], [10, 20, 30, 40])
        self.assertAlmostEqual(log["latitude"], 37.77, places=2)

    def test_record_legacy_dict_with_key_variation(self):
        """Verifies both 'anomaly_id' and 'id', 'anomaly_type' and 'type' keys are handled."""
        legacy_dict = {
            "id": 99,
            "type": "Water Stress",
            "confidence": 0.75,
            "score": 0.03,
            "bbox": [5, 5, 20, 20],
        }

        self.telemetry.record(legacy_dict)
        self.telemetry.flush()

        logs = self.telemetry.get_recent_logs(limit=1)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["anomaly_id"], 99)
        self.assertEqual(logs[0]["type"], "Water Stress")

    def test_synchronization_toggle(self):
        """Verifies marking records as synchronized upon ground station connection."""
        anom = CanopyAnomaly(
            anomaly_id=1,
            anomaly_type="Fungal Disease",
            confidence=0.9,
            mse_score=0.05,
            bbox=(0, 0, 10, 10),
            area_pixels=100,
        )
        self.telemetry.record(anom)
        self.assertEqual(self.telemetry.get_pending_sync_count(), 1)

        synced = self.telemetry.sync_logs_to_ground_station()
        self.assertEqual(synced, 1)
        self.assertEqual(self.telemetry.get_pending_sync_count(), 0)

    def test_sharp_backwards_compatible_alias(self):
        """Ensures existing SHARPOfflineResilienceManager imports function seamlessly."""
        alias_mgr = SHARPOfflineResilienceManager(db_path=self.db_path, auto_start_monitor=False)
        alias_mgr.log_threat_offline({"id": 7, "type": "Threat", "score": 0.1, "bbox": [0,0,1,1]})
        self.assertEqual(alias_mgr.get_pending_sync_count(), 1)


if __name__ == "__main__":
    unittest.main()
