"""
edge_node.py
============
Edge inference orchestrator for the physical Jetson Orin Nano environment.

This node acts as the main entrypoint. It integrates:
 1. Sensor ingest (mocked via a local directory stream for this iteration)
 2. End-to-end anomaly inspection pipeline
 3. Offline telemetry buffering and GIS Prescription Map export
"""

import os
import time
import glob
import logging
import argparse
from typing import Iterator

import cv2

# Adjust path so we can import sibling deployment modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from canopy_inspection import CanopyInspectionModule
from flight_telemetry import FlightTelemetrySpooler
from georeferencing import CameraIntrinsics, DroneTelemetry
from gis_export import generate_prescription_map, export_geojson

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def image_streamer(directory: str, fps_target: float) -> Iterator[str]:
    """Yields images from a directory at a simulated frame rate."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, ext)))
    files.sort()
    
    if not files:
        logging.error(f"No images found in stream directory {directory}")
        return

    delay = 1.0 / fps_target
    for f in files:
        yield f
        time.sleep(delay)

def run_edge_node(args):
    logging.info("Initializing Jetson Edge Orchestrator...")
    
    # 1. Initialize core inspection pipeline
    # The pipeline internally defaults to PyTorch backend but can be configured to load ONNX
    inspection_module = CanopyInspectionModule()
    
    # 2. Initialize flight telemetry spooler
    spooler = FlightTelemetrySpooler(db_path=args.db_path, auto_start_monitor=True)
    
    # 3. Static camera intrinsics (example defaults)
    intrinsics = CameraIntrinsics(
        image_width=args.img_width,
        image_height=args.img_height,
        hfov_deg=60.0,
        vfov_deg=45.0
    )
    
    # Mock starting telemetry
    current_lat = 40.7128
    current_lon = -74.0060
    altitude = 15.0 # meters
    
    all_anomalies = []
    frame_count = 0
    start_time = time.perf_counter()
    
    logging.info(f"Starting simulated flight loop targeting {args.fps} FPS")
    
    for img_path in image_streamer(args.input_stream, args.fps):
        # Update mock drone position slowly
        current_lat += 0.000001
        current_lon += 0.000001
        
        telemetry = DroneTelemetry(
            latitude=current_lat,
            longitude=current_lon,
            altitude_m=altitude,
            pitch_deg=0.0,
            roll_deg=0.0,
            yaw_deg=0.0
        )
        
        # Load frame (using OpenCV, though sensor_intake can handle path dicts/arrays directly)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        frame = cv2.resize(frame, (args.img_width, args.img_height))
        
        # Run inference
        result = inspection_module.inspect_with_diagnostics(frame, telemetry, intrinsics)
        
        # Spool anomalies to edge database
        for anomaly in result.anomalies:
            spooler.record(anomaly)
            all_anomalies.append(anomaly)
            
        frame_count += 1
        
        if frame_count % 10 == 0:
            logging.info(f"Processed {frame_count} frames... {len(result.anomalies)} anomalies detected in current frame.")

    elapsed = time.perf_counter() - start_time
    logging.info(f"Mission Complete. Processed {frame_count} frames in {elapsed:.2f}s ({frame_count/max(1e-5, elapsed):.1f} FPS)")
    
    # Flush remaining SQLite records
    spooler.flush()
    logging.info(f"Buffered anomalies successfully synced to {args.db_path}")
    
    # Export final GIS artifacts
    if all_anomalies:
        rates = {
            "Fungal/Bacterial Disease": 2.5,  # L/ha
            "Pest Infestation": 1.5,
            "Weed Invasion": 3.0,
            "Water Stress": 0.0,
            "Healthy Canopy": 0.0,
            "Unclassified Crop Stress": 1.0
        }
        rx_map = generate_prescription_map(all_anomalies, treatment_rates=rates, default_rate=0.0, flight_id="MIS-001")
        
        import json
        with open(args.export_geojson, "w") as f:
            json.dump(rx_map, f, indent=2)
        logging.info(f"Exported Variable-Rate Prescription map to {args.export_geojson}")
    else:
        logging.info("No anomalies detected. Skipping GIS export.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-stream", type=str, default="data/ucmerced_prepared", help="Directory acting as camera stream")
    parser.add_argument("--db-path", type=str, default="exports/flight_telemetry.db", help="SQLite spool path")
    parser.add_argument("--export-geojson", type=str, default="exports/prescription_map.geojson", help="Output VRA map path")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS for simulation")
    parser.add_argument("--img-width", type=int, default=640)
    parser.add_argument("--img-height", type=int, default=480)
    args = parser.parse_args()
    
    run_edge_node(args)
