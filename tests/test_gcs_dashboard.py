import os
import sqlite3
import json
import tempfile
import unittest
import pandas as pd

from gcs_dashboard import load_telemetry_data, load_geojson_map

class TestGCSDashboard(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for our mock database and geojson
        self.test_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.test_dir.name, "test_telemetry.db")
        self.geojson_path = os.path.join(self.test_dir.name, "test_map.geojson")
        
        # Setup mock SQLite DB
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE anomaly_logs (
                id INTEGER PRIMARY KEY,
                timestamp REAL,
                anomaly_id INTEGER,
                anomaly_type TEXT,
                confidence REAL,
                mse_score REAL,
                latitude REAL,
                longitude REAL,
                altitude REAL,
                estimated_area_sqm REAL
            )
        """)
        cur.execute("""
            INSERT INTO anomaly_logs (timestamp, anomaly_id, anomaly_type, confidence, mse_score, latitude, longitude, altitude, estimated_area_sqm)
            VALUES (1700000000.0, 1, 'Rust', 0.95, 0.05, 45.0, -120.0, 15.0, 2.5)
        """)
        conn.commit()
        conn.close()
        
        # Setup mock GeoJSON
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "id": 1,
                        "target_type": "Rust",
                        "application_rate_L_ha": 2.5,
                        "area_sqm": 2.5
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-120.0, 45.0], [-120.01, 45.0], [-120.0, 45.01], [-120.0, 45.0]]]
                    }
                }
            ]
        }
        with open(self.geojson_path, "w") as f:
            json.dump(geojson_data, f)
            
    def tearDown(self):
        self.test_dir.cleanup()
        
    def test_load_telemetry_data_success(self):
        df = load_telemetry_data(self.db_path)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["anomaly_type"], "Rust")
        self.assertIn("datetime", df.columns)
        
    def test_load_telemetry_data_missing_file(self):
        df = load_telemetry_data("nonexistent.db")
        self.assertTrue(df.empty)
        
    def test_load_geojson_map_success(self):
        data = load_geojson_map(self.geojson_path)
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertEqual(len(data["features"]), 1)
        self.assertEqual(data["features"][0]["properties"]["target_type"], "Rust")
        
    def test_load_geojson_map_missing_file(self):
        data = load_geojson_map("nonexistent.geojson")
        self.assertEqual(data, {})

if __name__ == "__main__":
    unittest.main()
