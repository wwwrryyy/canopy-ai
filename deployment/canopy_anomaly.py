"""
canopy_anomaly.py
=================
Domain model representing an agricultural anomaly region detected in aerial imagery.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional


@dataclass(frozen=True)
class CanopyAnomaly:
    """
    A localized vegetation region exhibiting significant spectral divergence from
    the healthy crop baseline (e.g. disease, pest damage, irrigation failure, weed invasion).
    """
    anomaly_id: int
    anomaly_type: str
    confidence: float
    mse_score: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in sensor pixels
    area_pixels: int
    
    # Milestone 4 Geospatial Additions
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude_m: Optional[float] = None
    estimated_area_sqm: Optional[float] = None
    polygon_wgs84: Optional[List[Tuple[float, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the anomaly to a dictionary with canonical and compatibility keys."""
        data = {
            "anomaly_id": self.anomaly_id,
            "id": self.anomaly_id,
            "anomaly_type": self.anomaly_type,
            "type": self.anomaly_type,
            "confidence": self.confidence,
            "mse_score": self.mse_score,
            "score": self.mse_score,
            "bbox": list(self.bbox),
            "area_pixels": self.area_pixels,
        }
        
        if self.gps_latitude is not None:
            data["gps_latitude"] = self.gps_latitude
        if self.gps_longitude is not None:
            data["gps_longitude"] = self.gps_longitude
        if self.gps_altitude_m is not None:
            data["gps_altitude_m"] = self.gps_altitude_m
        if self.estimated_area_sqm is not None:
            data["estimated_area_sqm"] = self.estimated_area_sqm
        if self.polygon_wgs84 is not None:
            data["polygon_wgs84"] = self.polygon_wgs84
            
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanopyAnomaly":
        """Constructs a CanopyAnomaly from a dictionary, handling key variations gracefully."""
        anomaly_id = data.get("anomaly_id", data.get("id", 0))
        anomaly_type = data.get("anomaly_type", data.get("type", "UNCLASSIFIED_ANOMALY"))
        confidence = float(data.get("confidence", 0.0))
        mse_score = float(data.get("mse_score", data.get("score", 0.0)))
        raw_bbox = data.get("bbox", [0, 0, 0, 0])
        bbox = (int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3]))
        area_pixels = int(data.get("area_pixels", bbox[2] * bbox[3]))

        return cls(
            anomaly_id=anomaly_id,
            anomaly_type=anomaly_type,
            confidence=confidence,
            mse_score=mse_score,
            bbox=bbox,
            area_pixels=area_pixels,
            gps_latitude=data.get("gps_latitude"),
            gps_longitude=data.get("gps_longitude"),
            gps_altitude_m=data.get("gps_altitude_m"),
            estimated_area_sqm=data.get("estimated_area_sqm"),
            polygon_wgs84=data.get("polygon_wgs84"),
        )
