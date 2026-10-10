import unittest
import math
from deployment.georeferencing import CameraIntrinsics, DroneTelemetry, compute_gsd, pixel_to_wgs84, bbox_to_polygon_wgs84, compute_area_sqm

class TestGeoreferencing(unittest.TestCase):
    def test_compute_gsd(self):
        # 100m altitude, 4000x3000 image, 60 deg HFOV, 45 deg VFOV
        gsd_x, gsd_y = compute_gsd(100.0, 4000, 3000, 60.0, 45.0)
        self.assertTrue(gsd_x > 0)
        self.assertTrue(gsd_y > 0)
        # Expected GSD: 
        # width = 2 * 100 * tan(30) = 200 * 0.577 = 115.47m
        # gsd_x = 115.47 / 4000 = ~0.0288 m/px
        self.assertTrue(math.isclose(gsd_x, 0.028867, rel_tol=1e-3))

    def test_pixel_to_wgs84(self):
        intrinsics = CameraIntrinsics(4000, 3000, 60.0, 45.0)
        # Right above the equator, 0 lon
        telemetry = DroneTelemetry(latitude=0.0, longitude=0.0, altitude_m=100.0, pitch_deg=0.0, roll_deg=0.0, yaw_deg=0.0)
        
        # Center pixel should map to the same GPS coordinate (nadir)
        lat, lon = pixel_to_wgs84(2000, 1500, intrinsics, telemetry)
        self.assertTrue(math.isclose(lat, 0.0, abs_tol=1e-7))
        self.assertTrue(math.isclose(lon, 0.0, abs_tol=1e-7))

    def test_bbox_to_polygon_wgs84(self):
        intrinsics = CameraIntrinsics(4000, 3000, 60.0, 45.0)
        telemetry = DroneTelemetry(latitude=45.0, longitude=-120.0, altitude_m=50.0, pitch_deg=0.0, roll_deg=0.0, yaw_deg=90.0)
        
        polygon = bbox_to_polygon_wgs84((1000, 1000, 500, 500), intrinsics, telemetry)
        self.assertEqual(len(polygon), 5)
        self.assertEqual(polygon[0], polygon[-1])

    def test_compute_area_sqm(self):
        intrinsics = CameraIntrinsics(4000, 3000, 60.0, 45.0)
        telemetry = DroneTelemetry(latitude=0.0, longitude=0.0, altitude_m=100.0, pitch_deg=0.0, roll_deg=0.0, yaw_deg=0.0)
        
        area = compute_area_sqm((0, 0, 100, 100), intrinsics, telemetry)
        self.assertTrue(area > 0)

