# Architecture Deepening Session Summary: `precision_agriculture`

- **Date**: October 10, 2026
- **Repository**: [`precision_agriculture`](https://github.com/wwwrryyy/precision_agriculture) (`origin/main`)
- **Key Commits**:
  - `c03dfca` — `feat(architecture): deepen canopy inspection pipeline and decouple edge seams`
  - `91a0311` — `feat(sensor-intake): deepen sensor ingestion and dynamic multispectral synthesis`
  - `6197602` — `docs: add architecture deepening session summary for reference`

---

## 1. Key Problems Discovered & Resolved

1. **Pipeline Fragmentation**:
   - **Previous state**: Callers in `drone_deployment_validation.py` had to coordinate 5 separate modules across 100+ lines of orchestration code: manually unpacking 4D tensors, computing difference heatmaps, slicing sub-crops, replicating 1-channel arrays 3×, and manually computing softmax.
   - **Fix**: Collapsed into [`CanopyInspectionModule`](deployment/canopy_inspection.py): a single deep interface `inspect(payload) -> list[CanopyAnomaly]`.

2. **Critical Telemetry Key Mismatch (Data Loss Bug)**:
   - **Previous state**: The pipeline emitted keys `{"anomaly_id", "anomaly_type", "confidence", "mse_score", "bbox"}`, while the SQLite persistence method in `sharp_sync_daemon.py` expected `{"id", "type", "score"}`. All logged anomalies were silently stored as ID `0`, type `UNCLASSIFIED_ANOMALY`, and score `0.0`.
   - **Fix**: Created [`FlightTelemetrySpooler`](deployment/flight_telemetry.py) accepting `CanopyAnomaly` domain objects directly, ensuring correct persistence of pathology classes, calibrated confidences, and coordinates.

3. **Sensor Disconnect & Channel Tripling**:
   - **Previous state**: Radiometric math (`compute_vegetation_indices.py`) was marooned in offline data prep, while the edge runtime assumed 1-channel grayscale and artificially replicated channels.
   - **Fix**: Built [`SensorIntakeModule`](deployment/sensor_intake.py), automatically adapting 5-band multispectral, 3-band RGB, or 1-band SAR feeds, synthesizing calibrated **NDVI** and **NDRE** indices, and establishing a single source of truth for radiometric formulas.

4. **SNN Phase-Out Clarification**:
   - Clarified that earlier Spiking Neural Networks (`SpikingUNet` with $T=8$ LIF neurons) were phased out in favor of single-pass ANNs (U-Net, ConvAutoencoder, MobileNetV3-Small) to eliminate the 4×–8× latency penalty and enable native TensorRT execution on the **NVIDIA Jetson Orin Nano**.

---

## 2. Architecture & File Reference

| Module / Seam | File Path | Key Role / Interface |
|---|---|---|
| **Domain Glossary** | [`GLOSSARY.md`](GLOSSARY.md) | Canonical domain terms (*Canopy Frame*, *Multispectral Canopy Frame*, *Canopy Anomaly*, etc.) |
| **Domain Model** | [`deployment/canopy_anomaly.py`](deployment/canopy_anomaly.py) | Strongly-typed `CanopyAnomaly` dataclass with `.to_dict()` and `.from_dict()` |
| **Canopy Inspection** | [`deployment/canopy_inspection.py`](deployment/canopy_inspection.py) | Deep module: `inspect(payload) -> list[CanopyAnomaly]` |
| **Sensor Intake** | [`deployment/sensor_intake.py`](deployment/sensor_intake.py) | Ingests multi-band payloads, computes NDVI/NDRE, provides `MultispectralCanopyFrame` |
| **Inference Seam** | [`deployment/inference_backend.py`](deployment/inference_backend.py) | Swappable `OnnxInferenceBackend`, `PyTorchInferenceBackend`, and `MockInferenceBackend` |
| **Difference Seam** | [`deployment/difference_engine.py`](deployment/difference_engine.py) | `NativeDifferenceAdapter` (C++) and `OpenCVDifferenceAdapter` (NumPy/cv2) |
| **Flight Telemetry** | [`deployment/flight_telemetry.py`](deployment/flight_telemetry.py) | Memory-buffered spooler with single-transaction SQLite batch persistence |
| **Legacy Compatibility** | [`deployment/sharp_sync_daemon.py`](deployment/sharp_sync_daemon.py) | Subclasses `FlightTelemetrySpooler` for 100% backward compatibility |
| **Offline Radiometry** | [`data_prep/compute_vegetation_indices.py`](data_prep/compute_vegetation_indices.py) | Delegates to `sensor_intake` for consistent NDVI/NDRE math |

---

## 3. Verification Metrics

- **Automated Tests**: [`tests/`](tests/) contains **14 tests across 3 suites**, all passing in **0.015s**:
  ```bash
  python -m unittest discover -s tests -p "test_*.py"
  ```
- **Edge Deployment Benchmark**:
  ```bash
  python deployment/drone_deployment_validation.py --frames 15
  ```
  - **Throughput**: **330.27 FPS** (exceeds the $\ge 10.0$ FPS real-time target by 33×)
  - **Latency**: Mean 3.03 ms / P50 2.47 ms
  - **Telemetry Check**: Verified actual classified anomalies (`ID: 1`, `Type: Healthy Canopy`, `Conf: 0.20`, `MSE: 0.0011`) stored in SQLite.
- **CodeRabbit Review**: Passed with **0 findings**.

---

## 4. Suggested Starting Points for Next Session

When you return to this project, here are the three highest-value directions to pick up:

1. **Stage 2 Anomaly Detection Upgrade**:
   - Evolve the pixel-subtraction autoencoder to a **Student-Teacher Feature Pyramid (STPM)** using MobileNetV3 feature embeddings, or augment the C++ difference engine with **SSIM/LNCC** filtering to eliminate cloud shadow false alarms.
2. **Cloud Telemetry Egress Adapter**:
   - Implement an egress sync adapter in `FlightTelemetrySpooler` for live auto-sync with John Deere Operations Center REST API or MQTT ground station telemetry brokers.
3. **Jetson Orin Nano Native C++ / TensorRT Bridge**:
   - Modernize the CMake build configuration in `cpp_extensions/anomaly_calculus/CMakeLists.txt` to compile native TensorRT pybind execution engines across Python environments.
