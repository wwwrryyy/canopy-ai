"""
spatial_clustering.py
=====================
Deep module for spatial anomaly clustering.
Filters spatial MSE heatmaps, extracts contours, and applies Non-Maximum Suppression (NMS).
"""

from typing import List, Tuple
import cv2
import numpy as np

class SpatialClusteringEngine:
    """
    Filters spatial MSE anomaly heatmaps, runs contour clustering, and applies NMS.
    """
    def __init__(self, mse_threshold: float = 0.001, min_area: int = 20, max_area: int = 15000, nms_iou_threshold: float = 0.3):
        self.mse_threshold = mse_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.nms_iou_threshold = nms_iou_threshold

    def extract_anomaly_regions(self, heatmap: np.ndarray) -> Tuple[List[List[int]], List[float]]:
        """
        Isolates high-error contours and applies NMS to return unique anomaly regions.
        Args:
            heatmap: 2D NumPy array of pixel-wise MSE errors.
        Returns:
            Tuple of (nms_boxes, nms_scores)
        """
        if heatmap is None or heatmap.size == 0:
            return [], []

        binary_mask = (heatmap > self.mse_threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []
        candidate_scores = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = heatmap[y:y+h, x:x+w]
                score = float(np.mean(roi)) if roi.size > 0 else float(self.mse_threshold)
                candidate_boxes.append([x, y, w, h])
                candidate_scores.append(score)

        return self._apply_nms(candidate_boxes, candidate_scores)

    def _apply_nms(self, boxes: List[List[int]], scores: List[float]) -> Tuple[List[List[int]], List[float]]:
        if not boxes:
            return [], []

        boxes_arr = np.array(boxes)
        scores_arr = np.array(scores)
        idxs = np.argsort(scores_arr)[::-1]
        keep = []

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

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = box_a[2] * box_a[3]
        boxBArea = box_b[2] * box_b[3]

        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
