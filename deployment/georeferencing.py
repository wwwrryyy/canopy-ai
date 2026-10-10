"""
georeferencing.py
=================
Engine for projecting 2D pixel coordinates to 3D WGS84 GPS coordinates using a pinhole camera model.
"""

import math
from dataclasses import dataclass
from typing import Tuple, List

# WGS84 ellipsoid constants
WGS84_A = 6378137.0  # Semi-major axis in meters
WGS84_E2 = 0.00669437999014  # First eccentricity squared

@dataclass
class CameraIntrinsics:
    image_width: int
    image_height: int
    hfov_deg: float
    vfov_deg: float

@dataclass
class DroneTelemetry:
    latitude: float
    longitude: float
    altitude_m: float  # Above ground level
    pitch_deg: float
    roll_deg: float
    yaw_deg: float

def compute_gsd(altitude_m: float, image_width: int, image_height: int, hfov_deg: float, vfov_deg: float) -> Tuple[float, float]:
    """Computes Ground Sampling Distance (GSD) in meters per pixel."""
    hfov_rad = math.radians(hfov_deg)
    vfov_rad = math.radians(vfov_deg)
    
    # Ground width/height in meters assuming nadir
    ground_width = 2 * altitude_m * math.tan(hfov_rad / 2)
    ground_height = 2 * altitude_m * math.tan(vfov_rad / 2)
    
    gsd_x = ground_width / image_width
    gsd_y = ground_height / image_height
    
    return gsd_x, gsd_y

def offset_wgs84(lat: float, lon: float, offset_east_m: float, offset_north_m: float) -> Tuple[float, float]:
    """Adds a local meter offset to WGS84 coordinates."""
    lat_rad = math.radians(lat)
    
    # Radii of curvature
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_rad)**2)
    M = (WGS84_A * (1 - WGS84_E2)) / ((1 - WGS84_E2 * math.sin(lat_rad)**2)**1.5)
    
    delta_lat = offset_north_m / M
    delta_lon = offset_east_m / (N * math.cos(lat_rad))
    
    new_lat = lat + math.degrees(delta_lat)
    new_lon = lon + math.degrees(delta_lon)
    
    return new_lat, new_lon

def pixel_to_wgs84(
    pixel_x: float, 
    pixel_y: float, 
    intrinsics: CameraIntrinsics, 
    telemetry: DroneTelemetry
) -> Tuple[float, float]:
    """Projects a single pixel to WGS84 coordinates."""
    gsd_x, gsd_y = compute_gsd(
        telemetry.altitude_m, 
        intrinsics.image_width, 
        intrinsics.image_height, 
        intrinsics.hfov_deg, 
        intrinsics.vfov_deg
    )
    
    # Principal point (center of image)
    cx = intrinsics.image_width / 2.0
    cy = intrinsics.image_height / 2.0
    
    # Pixel offset from center
    # Image coords: top-left is 0,0. Positive dy_px is "forward"
    dx_px = pixel_x - cx
    dy_px = cy - pixel_y 
    
    # Offset in meters on the ground (assuming nadir for now, basic pinhole)
    # To properly handle pitch and roll, we would use a rotation matrix and intersect with a ground plane,
    # but for typical agricultural drone flights (nadir), simple GSD with yaw is sufficient for Milestone 4.
    offset_x_m = dx_px * gsd_x
    offset_y_m = dy_px * gsd_y
    
    yaw_rad = math.radians(telemetry.yaw_deg)
    
    # Rotate offset by yaw (0 degrees = North, 90 = East)
    # We rotate the camera frame to the world frame (North, East)
    east_m = offset_x_m * math.cos(yaw_rad) + offset_y_m * math.sin(yaw_rad)
    north_m = -offset_x_m * math.sin(yaw_rad) + offset_y_m * math.cos(yaw_rad)
    
    return offset_wgs84(telemetry.latitude, telemetry.longitude, east_m, north_m)

def bbox_to_polygon_wgs84(
    bbox: Tuple[int, int, int, int], 
    intrinsics: CameraIntrinsics, 
    telemetry: DroneTelemetry
) -> List[Tuple[float, float]]:
    """Converts a bounding box (x, y, w, h) into a WGS84 polygon."""
    x, y, w, h = bbox
    
    # Four corners
    corners = [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h)
    ]
    
    polygon = []
    for px, py in corners:
        polygon.append(pixel_to_wgs84(px, py, intrinsics, telemetry))
        
    # Close the polygon
    polygon.append(polygon[0])
    return polygon

def compute_area_sqm(bbox: Tuple[int, int, int, int], intrinsics: CameraIntrinsics, telemetry: DroneTelemetry) -> float:
    """Estimates the area of a bounding box in square meters."""
    gsd_x, gsd_y = compute_gsd(
        telemetry.altitude_m, 
        intrinsics.image_width, 
        intrinsics.image_height, 
        intrinsics.hfov_deg, 
        intrinsics.vfov_deg
    )
    _, _, w, h = bbox
    return (w * gsd_x) * (h * gsd_y)

@dataclass
class GeolocationRecord:
    gps_lat: float
    gps_lon: float
    gps_alt: float
    area_sqm: float
    polygon_wgs84: List[Tuple[float, float]]

class GeoreferencingEngine:
    """
    Deep module interface encapsulating all GIS projection logic.
    Converts 2D pixel bounding boxes into 3D WGS84 records.
    """
    def project_anomaly(
        self,
        bbox: Tuple[int, int, int, int],
        intrinsics: CameraIntrinsics,
        telemetry: DroneTelemetry
    ) -> GeolocationRecord:
        x, y, w, h = bbox
        
        area_sqm = compute_area_sqm(bbox, intrinsics, telemetry)
        polygon = bbox_to_polygon_wgs84(bbox, intrinsics, telemetry)
        
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        gps_lat, gps_lon = pixel_to_wgs84(center_x, center_y, intrinsics, telemetry)
        
        return GeolocationRecord(
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            gps_alt=telemetry.altitude_m,
            area_sqm=area_sqm,
            polygon_wgs84=polygon
        )
