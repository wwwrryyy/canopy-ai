import unittest
from deployment.canopy_anomaly import CanopyAnomaly
from deployment.gis_export import export_geojson, generate_prescription_map

class TestGisExport(unittest.TestCase):
    def test_export_geojson(self):
        anomaly1 = CanopyAnomaly(
            anomaly_id=1,
            anomaly_type="Rust",
            confidence=0.9,
            mse_score=0.05,
            bbox=(0, 0, 10, 10),
            area_pixels=100,
            gps_latitude=45.0,
            gps_longitude=-120.0,
            gps_altitude_m=10.0,
            estimated_area_sqm=5.0,
            polygon_wgs84=[(-120.0, 45.0), (-120.01, 45.0), (-120.01, 45.01), (-120.0, 45.01), (-120.0, 45.0)]
        )
        
        anomaly2 = CanopyAnomaly(
            anomaly_id=2,
            anomaly_type="Healthy",
            confidence=0.8,
            mse_score=0.01,
            bbox=(50, 50, 10, 10),
            area_pixels=100,
            # No polygon, should be skipped
        )
        
        geojson = export_geojson([anomaly1, anomaly2], flight_id="flight123")
        
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(geojson["metadata"]["flight_id"], "flight123")
        self.assertEqual(len(geojson["features"]), 1)
        
        feature = geojson["features"][0]
        self.assertEqual(feature["geometry"]["type"], "Polygon")
        self.assertEqual(feature["properties"]["type"], "Rust")

    def test_generate_prescription_map(self):
        anomaly1 = CanopyAnomaly(
            anomaly_id=1,
            anomaly_type="Rust",
            confidence=0.9,
            mse_score=0.05,
            bbox=(0, 0, 10, 10),
            area_pixels=100,
            polygon_wgs84=[(-120.0, 45.0), (-120.01, 45.0), (-120.01, 45.01), (-120.0, 45.01), (-120.0, 45.0)]
        )
        
        rates = {"Rust": 2.5}
        
        rx_map = generate_prescription_map([anomaly1], treatment_rates=rates, default_rate=0.0)
        
        self.assertEqual(rx_map["type"], "FeatureCollection")
        self.assertEqual(len(rx_map["features"]), 1)
        self.assertEqual(rx_map["features"][0]["properties"]["application_rate_L_ha"], 2.5)
