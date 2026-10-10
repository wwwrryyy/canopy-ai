"""
gis_export.py
=============
Exports localized anomaly objects into standard GIS formats for agronomic integration.
Generates GeoJSON FeatureCollections and Variable-Rate Agronomic Prescription Maps.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from canopy_anomaly import CanopyAnomaly

def export_geojson(anomalies: List[CanopyAnomaly], flight_id: str = "unknown") -> Dict[str, Any]:
    """
    Converts a list of anomalies into a GeoJSON FeatureCollection.
    """
    features = []
    
    for anomaly in anomalies:
        if anomaly.polygon_wgs84 is None or len(anomaly.polygon_wgs84) == 0:
            continue
            
        # GeoJSON uses [longitude, latitude]
        coordinates = [[[lon, lat] for lat, lon in anomaly.polygon_wgs84]]
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates
            },
            "properties": {
                "id": anomaly.anomaly_id,
                "type": anomaly.anomaly_type,
                "confidence": anomaly.confidence,
                "score": anomaly.mse_score,
                "area_sqm": anomaly.estimated_area_sqm
            }
        }
        features.append(feature)
        
    return {
        "type": "FeatureCollection",
        "metadata": {
            "flight_id": flight_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "feature_count": len(features)
        },
        "features": features
    }

def generate_prescription_map(
    anomalies: List[CanopyAnomaly], 
    treatment_rates: Dict[str, float], 
    default_rate: float = 0.0,
    flight_id: str = "unknown"
) -> Dict[str, Any]:
    """
    Generates a Variable-Rate Prescription map in GeoJSON format.
    treatment_rates maps anomaly_type to a specific application rate (e.g. L/ha).
    """
    features = []
    
    for anomaly in anomalies:
        if anomaly.polygon_wgs84 is None or len(anomaly.polygon_wgs84) == 0:
            continue
            
        rate = treatment_rates.get(anomaly.anomaly_type, default_rate)
        if rate <= 0.0:
            continue
            
        coordinates = [[[lon, lat] for lat, lon in anomaly.polygon_wgs84]]
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates
            },
            "properties": {
                "id": anomaly.anomaly_id,
                "target_type": anomaly.anomaly_type,
                "application_rate_L_ha": rate,
                "area_sqm": anomaly.estimated_area_sqm
            }
        }
        features.append(feature)
        
    return {
        "type": "FeatureCollection",
        "metadata": {
            "flight_id": flight_id,
            "map_type": "VariableRatePrescription",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "feature_count": len(features)
        },
        "features": features
    }
