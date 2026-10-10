"""
canopy_inspection.py
====================
Deep module consolidating the complete aerial precision agriculture pipeline:
  0. Sensor intake, radiance calibration & multispectral vegetation index synthesis
  1. Despeckling image restoration
  2. Baseline healthy canopy reconstruction & difference heatmap calculation
  3. Spatial contour extraction & Non-Maximum Suppression (NMS)
  4. Localized anomaly sub-crop extraction, multi-channel projection, and pathology classification

All tensor conversions, sub-crop slicing, multi-channel expansions, and stage handoffs
are encapsulated behind a single, high-leverage interface:
  inspect(payload) -> list[CanopyAnomaly]
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from canopy_anomaly import CanopyAnomaly
from difference_engine import DifferenceConfig, DifferenceEngine, create_difference_engine
from inference_backend import InferenceBackend, create_inference_backend
from sensor_intake import SensorIntakeModule, MultispectralCanopyFrame
from georeferencing import DroneTelemetry, CameraIntrinsics, compute_area_sqm, bbox_to_polygon_wgs84


@dataclass
class CanopyInspectionResult:
    """Diagnostic result bundle useful for ground stations and latency benchmarking."""
    anomalies: List[CanopyAnomaly]
    canopy_frame: MultispectralCanopyFrame
    clean_frame: np.ndarray
    recon_frame: np.ndarray
    heatmap: np.ndarray
    latencies_ms: Dict[str, float]


class CanopyInspectionModule:
    """
    Deep module providing end-to-end canopy anomaly detection and classification.
    Accepts raw multispectral feeds, RGB images, SAR radar arrays, or band dictionaries.
    """

    def __init__(
        self,
        inference_backend: Optional[InferenceBackend] = None,
        difference_engine: Optional[DifferenceEngine] = None,
        difference_config: Optional[DifferenceConfig] = None,
        sensor_intake: Optional[SensorIntakeModule] = None,
        mse_threshold: float = 0.001,
        min_area: int = 20,
        max_area: int = 15000,
        nms_iou_threshold: float = 0.3,
        gaussian_sigma: float = 1.5,
        min_confidence: float = 0.50,
    ):
        self.difference_config = difference_config
        self.backend = inference_backend or create_inference_backend()
        self.diff_engine = difference_engine or create_difference_engine(config=self.difference_config)
        self.sensor_intake = sensor_intake or SensorIntakeModule(target_size=(128, 128))
        self.mse_threshold = mse_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.nms_iou_threshold = nms_iou_threshold
        self.gaussian_sigma = gaussian_sigma
        self.min_confidence = min_confidence

    def inspect(
        self,
        payload: Union[np.ndarray, Dict[str, Union[np.ndarray, str]], MultispectralCanopyFrame],
        telemetry: Optional[DroneTelemetry] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> List[CanopyAnomaly]:
        """
        Primary interface: processes sensor payload and returns all detected anomalies.

        Args:
            payload: MultispectralCanopyFrame, 2D/3D numpy array, or band path dict.
            telemetry: Optional flight telemetry for georeferencing.
            intrinsics: Optional camera intrinsics for georeferencing.

        Returns:
            List of detected and classified CanopyAnomaly domain objects.
        """
        result = self.inspect_with_diagnostics(payload, telemetry, intrinsics)
        return result.anomalies

    def inspect_with_diagnostics(
        self,
        payload: Union[np.ndarray, Dict[str, Union[np.ndarray, str]], MultispectralCanopyFrame],
        telemetry: Optional[DroneTelemetry] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> CanopyInspectionResult:
        """
        Diagnostic interface: processes sensor feed and returns anomalies with intermediate maps and latencies.
        """
        t0 = time.perf_counter()

        # 0. Stage 0: Sensor Ingestion & Multispectral Synthesis
        canopy_frame = self.sensor_intake.ingest(payload)
        frame_128 = canopy_frame.to_single_channel()
        composite_3ch = canopy_frame.to_rgb_composite()
        t1 = time.perf_counter()

        # 1. Stage 1: Despeckling image restoration
        clean_2d = self.backend.despeckle(frame_128)
        t2 = time.perf_counter()

        # 2. Stage 2: Healthy canopy baseline reconstruction
        recon_2d = self.backend.reconstruct_baseline(clean_2d)
        t3 = time.perf_counter()

        # 3. Difference heatmap computation
        heatmap = self.diff_engine.compute_heatmap(
            clean_2d, recon_2d, sigma=self.gaussian_sigma, config=self.difference_config
        )
        t4 = time.perf_counter()

        # 4. Spatial contour extraction and NMS clustering
        candidate_boxes, candidate_scores = self._extract_spatial_candidates(heatmap)
        nms_boxes, nms_scores = self._apply_nms(candidate_boxes, candidate_scores)
        t5 = time.perf_counter()

        # 5. Stage 3: Multi-channel sub-crop extraction and pathology classification
        anomalies: List[CanopyAnomaly] = []
        for i, (box, score) in enumerate(zip(nms_boxes, nms_scores)):
            x, y, w, h = box
            frame_h, frame_w = composite_3ch.shape[1], composite_3ch.shape[2]
            y_min, y_max = max(0, y), min(frame_h, y + h)
            x_min, x_max = max(0, x), min(frame_w, x + w)

            # Crop from 3-channel spectral composite [3, H, W]
            crop_3ch = composite_3ch[:, y_min:y_max, x_min:x_max]
            if crop_3ch.size == 0 or crop_3ch.shape[1] == 0 or crop_3ch.shape[2] == 0:
                continue

            # Standardize crop patch to 64x64
            crop_resized = np.zeros((3, 64, 64), dtype=np.float32)
            for c_idx in range(3):
                crop_resized[c_idx] = cv2.resize(crop_3ch[c_idx], (64, 64))

            class_name, confidence = self.backend.classify_crop(crop_resized)
            effective_class = (
                class_name if confidence >= self.min_confidence else "Unclassified Crop Stress"
            )

            # Optional georeferencing if telemetry is provided
            gps_lat, gps_lon, gps_alt, area_sqm, polygon_wgs84 = None, None, None, None, None
            if telemetry is not None and intrinsics is not None:
                gps_alt = telemetry.altitude_m
                clipped_box = (x_min, y_min, x_max - x_min, y_max - y_min)
                area_sqm = compute_area_sqm(clipped_box, intrinsics, telemetry)
                polygon_wgs84 = bbox_to_polygon_wgs84(clipped_box, intrinsics, telemetry)
                # We can estimate the center point as the GPS coordinate for the anomaly
                center_x, center_y = x_min + (x_max - x_min) / 2.0, y_min + (y_max - y_min) / 2.0
                from georeferencing import pixel_to_wgs84
                gps_lat, gps_lon = pixel_to_wgs84(center_x, center_y, intrinsics, telemetry)

            anomalies.append(
                CanopyAnomaly(
                    anomaly_id=i + 1,
                    anomaly_type=effective_class,
                    confidence=confidence,
                    mse_score=score,
                    bbox=(x, y, w, h),
                    area_pixels=w * h,
                    gps_latitude=gps_lat,
                    gps_longitude=gps_lon,
                    gps_altitude_m=gps_alt,
                    estimated_area_sqm=area_sqm,
                    polygon_wgs84=polygon_wgs84,
                )
            )
        t6 = time.perf_counter()

        latencies = {
            "stage0_sensor_intake": (t1 - t0) * 1000.0,
            "stage1_despeckle": (t2 - t1) * 1000.0,
            "stage2_autoencoder": (t3 - t2) * 1000.0,
            "heatmap_diff": (t4 - t3) * 1000.0,
            "spatial_nms": (t5 - t4) * 1000.0,
            "stage3_classifier": (t6 - t5) * 1000.0,
            "total": (t6 - t0) * 1000.0,
        }

        return CanopyInspectionResult(
            anomalies=anomalies,
            canopy_frame=canopy_frame,
            clean_frame=clean_2d,
            recon_frame=recon_2d,
            heatmap=heatmap,
            latencies_ms=latencies,
        )

    def _extract_spatial_candidates(
        self, heatmap: np.ndarray
    ) -> Tuple[List[List[int]], List[float]]:
        """Isolates high-error contours after morphological filtering."""
        if heatmap is None or heatmap.size == 0:
            return [], []

        binary_mask = (heatmap > self.mse_threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes: List[List[int]] = []
        candidate_scores: List[float] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = heatmap[y:y+h, x:x+w]
                score = float(np.mean(roi)) if roi.size > 0 else float(self.mse_threshold)
                candidate_boxes.append([x, y, w, h])
                candidate_scores.append(score)

        return candidate_boxes, candidate_scores

    def _apply_nms(
        self, boxes: List[List[int]], scores: List[float]
    ) -> Tuple[List[List[int]], List[float]]:
        """Suppresses overlapping bounding boxes by IoU."""
        if not boxes:
            return [], []

        boxes_arr = np.array(boxes)
        scores_arr = np.array(scores)
        idxs = np.argsort(scores_arr)[::-1]
        keep: List[int] = []

        while len(idxs) > 0:
            curr = idxs[0]
            keep.append(curr)
            if len(idxs) == 1:
                break

            rest = idxs[1:]
            filtered = []
            for i in rest:
                if self._compute_iou(boxes_arr[curr], boxes_arr[i]) < self.nms_iou_threshold:
                    filtered.append(i)
            idxs = np.array(filtered)

        return [boxes[i] for i in keep], [scores[i] for i in keep]

    @staticmethod
    def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
        yB = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        area_a = box_a[2] * box_a[3]
        area_b = box_b[2] * box_b[3]
        return inter / float(area_a + area_b - inter + 1e-6)
