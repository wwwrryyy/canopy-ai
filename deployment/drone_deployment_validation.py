"""
drone_deployment_validation.py
==============================
End-to-end validation and latency benchmarking for the 3-stage precision agriculture pipeline.
Exercised through the deep CanopyInspectionModule and FlightTelemetrySpooler interfaces.

Verifies >= 10 FPS throughput target for edge deployment on NVIDIA Jetson Orin Nano.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

# Ensure project root and deployment dirs are on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)

from canopy_anomaly import CanopyAnomaly
from difference_engine import create_difference_engine, HAS_AGRI_CORE
from inference_backend import create_inference_backend, HAS_ORT, HAS_TORCH
from canopy_inspection import CanopyInspectionModule
from flight_telemetry import FlightTelemetrySpooler


class EndToEndPipeline:
    """
    Validation harness delegating entirely to the deep CanopyInspectionModule.
    """

    def __init__(self, exports_dir=None, use_onnx=True, mse_threshold=0.001, min_area=20):
        if exports_dir is None:
            exports_dir = os.path.join(PROJECT_ROOT, "exports")
        self.exports_dir = exports_dir

        backend = create_inference_backend(exports_dir=exports_dir, prefer_onnx=use_onnx)
        diff_engine = create_difference_engine(prefer_native=True)

        self.module = CanopyInspectionModule(
            inference_backend=backend,
            difference_engine=diff_engine,
            mse_threshold=mse_threshold,
            min_area=min_area,
        )

    def process_frame(self, raw_frame_128):
        """
        Runs one 128x128 frame through the consolidated inspection module.
        """
        diag = self.module.inspect_with_diagnostics(raw_frame_128)
        return {
            "clean_frame": diag.clean_frame,
            "recon_frame": diag.recon_frame,
            "heatmap": diag.heatmap,
            "anomalies": diag.anomalies,
            "latencies_ms": diag.latencies_ms,
        }


def run_benchmark(num_frames=50, db_path="test_flight_telemetry.db"):
    print("=" * 70)
    print(" Precision Agriculture Drone Deployment Benchmark")
    print("=" * 70)
    print(f"[*] Framework: {'ONNX Runtime' if HAS_ORT else 'PyTorch'}")
    print(f"[*] Difference Engine: {'C++ agri_core' if HAS_AGRI_CORE else 'OpenCV fallback'}")

    pipeline = EndToEndPipeline()
    telemetry = FlightTelemetrySpooler(db_path=db_path)

    # Synthetic canopy frames with occasional anomalies
    print(f"[*] Simulating drone flight over {num_frames} canopy frames...")
    np.random.seed(42)

    lat_base = 36.7783
    lon_base = -119.4179
    altitude_m = 45.0

    times = []
    total_anomalies = 0

    for i in range(num_frames):
        # Base healthy crop canopy texture
        frame = np.random.uniform(0.3, 0.7, (128, 128)).astype(np.float32)
        # Inject artificial anomaly patch in some frames
        if i % 3 == 0:
            cx, cy = np.random.randint(30, 98), np.random.randint(30, 98)
            rad = np.random.randint(8, 18)
            cv2.circle(frame, (cx, cy), rad, float(np.random.choice([0.05, 0.95])), -1)

        result = pipeline.process_frame(frame)
        times.append(result["latencies_ms"]["total"])

        # Log anomalies to offline telemetry buffer with GPS
        for anom in result["anomalies"]:
            total_anomalies += 1
            telemetry.record(
                anom,
                latitude=lat_base + i * 0.0001,
                longitude=lon_base + i * 0.0001,
                altitude=altitude_m + float(np.random.normal(0, 0.2)),
            )

    telemetry.flush()

    times = np.array(times)
    mean_lat = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    fps = 1000.0 / mean_lat

    print("\n--- Benchmark Results ---")
    print(f"Frames processed:       {num_frames}")
    print(f"Mean Latency:           {mean_lat:.2f} ms")
    print(f"Median (P50) Latency:   {p50:.2f} ms")
    print(f"95th Percentile (P95):  {p95:.2f} ms")
    print(f"Throughput:             {fps:.2f} FPS")
    print(f"Anomalies Detected:     {total_anomalies}")

    # Check telemetry DB records
    logs = telemetry.get_recent_logs(limit=5)
    print(f"\n--- Telemetry Persistence Verification ---")
    print(f"Telemetry logs stored:  {len(logs)} samples verified in {db_path}")
    for idx, log in enumerate(logs):
        print(f"  [{idx + 1}] ID:{log['anomaly_id']} | Type:{log['type']} | Conf:{log['confidence']:.2f} | MSE:{log['mse_score']:.4f}")

    # Clean up test DB
    telemetry.stop_network_monitor()
    if os.path.exists(db_path):
        os.remove(db_path)

    print("\n--- Target Verification ---")
    if fps >= 10.0:
        print(f"[+] PASS: Pipeline achieves {fps:.2f} FPS >= 10.0 FPS real-time threshold!")
        return 0
    else:
        print(f"[-] WARNING: Pipeline achieves {fps:.2f} FPS < 10.0 FPS threshold.")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Deployment Validation")
    parser.add_argument("--frames", type=int, default=50, help="Number of test frames")
    args = parser.parse_args()
    sys.exit(run_benchmark(num_frames=args.frames))
