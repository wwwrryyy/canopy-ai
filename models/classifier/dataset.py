"""
dataset.py
==========
Agricultural pathology dataset for Stage 3 MobileNetV3 classifier.
Loads base canopy patches and synthesizes realistic foliar pathologies:
- Class 0: Healthy Canopy
- Class 1: Fungal/Bacterial Disease (necrotic spots with chlorotic halos)
- Class 2: Pest Infestation (chewed holes, skeletonization, bite stippling)
- Class 3: Water Stress (diffuse drought chlorosis, desiccation)
- Class 4: Weed Invasion (foreign vegetation clumping between crop rows)

Also supports loading real external datasets organized by subfolder (e.g., PlantVillage, CropDeep).
"""

import os
import glob
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from classifier import CLASS_NAMES

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class AgriculturalPathologyDataset(Dataset):
    """
    Dataset generating and loading 64x64 agricultural pathology patches.
    """

    def __init__(
        self,
        base_patch_dirs: Optional[List[str]] = None,
        external_data_dir: Optional[str] = None,
        samples_per_class: int = 400,
        target_size: Tuple[int, int] = (64, 64),
        is_train: bool = True,
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.target_size = target_size
        self.is_train = is_train
        self.samples_per_class = samples_per_class
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.external_images: List[Tuple[str, int]] = []
        if external_data_dir and os.path.isdir(external_data_dir):
            for class_idx, class_name in enumerate(CLASS_NAMES):
                folder = os.path.join(external_data_dir, class_name)
                if os.path.isdir(folder):
                    for ext in ("*.png", "*.jpg", "*.jpeg"):
                        for p in glob.glob(os.path.join(folder, ext)):
                            self.external_images.append((p, class_idx))

        # Base canopy patches for synthesis
        self.base_patch_paths: List[str] = []
        if not self.external_images:
            default_dirs = [
                os.path.join(os.path.dirname(__file__), "..", "..", "data", "ucmerced_prepared", "agricultural_patches"),
                os.path.join(os.path.dirname(__file__), "..", "..", "data", "patches"),
            ]
            dirs_to_search = base_patch_dirs or default_dirs
            for d in dirs_to_search:
                if os.path.isdir(d):
                    self.base_patch_paths.extend(glob.glob(os.path.join(d, "*.png")))
                    self.base_patch_paths.extend(glob.glob(os.path.join(d, "*.jpg")))

            self.base_patch_paths = sorted(self.base_patch_paths)

        self.num_classes = len(CLASS_NAMES)
        self.total_samples = len(self.external_images) if self.external_images else self.samples_per_class * self.num_classes

    def __len__(self) -> int:
        return self.total_samples

    def _load_base_patch(self, idx: int) -> np.ndarray:
        """Loads and crops a base canopy patch to target_size in RGB format [0, 1]."""
        if self.base_patch_paths:
            path = self.base_patch_paths[idx % len(self.base_patch_paths)]
            bgr = cv2.imread(path)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                # Random crop or resize to target_size
                h, w, _ = rgb.shape
                th, tw = self.target_size
                if h >= th and w >= tw:
                    y0 = random.randint(0, h - th)
                    x0 = random.randint(0, w - tw)
                    crop = rgb[y0:y0 + th, x0:x0 + tw]
                else:
                    crop = cv2.resize(rgb, (tw, th))
                return crop.astype(np.float32) / 255.0

        # Fallback procedural crop rows if no disk patches available
        th, tw = self.target_size
        x = np.linspace(0, 4 * np.pi, tw)
        rows = (np.sin(x)[None, :] * 0.15 + 0.55).astype(np.float32)
        base = np.repeat(rows, th, axis=0)
        texture = np.random.normal(0, 0.03, (th, tw)).astype(np.float32)
        green = np.clip(base + texture, 0.1, 0.9)
        # BGR / RGB synthetic canopy
        rgb = np.zeros((th, tw, 3), dtype=np.float32)
        rgb[:, :, 0] = green * 0.4  # Red
        rgb[:, :, 1] = green        # Green
        rgb[:, :, 2] = green * 0.3  # Blue
        return rgb

    def _synthesize_fungal_disease(self, base_rgb: np.ndarray) -> np.ndarray:
        """Injects necrotic spots surrounded by chlorotic yellow halos."""
        img = base_rgb.copy()
        th, tw, _ = img.shape
        num_spots = random.randint(3, 8)
        for _ in range(num_spots):
            cx = random.randint(8, tw - 9)
            cy = random.randint(8, th - 9)
            radius = random.randint(3, 7)

            # Chlorotic yellow halo (increased R & G, decreased B)
            halo_mask = np.zeros((th, tw), dtype=np.float32)
            cv2.circle(halo_mask, (cx, cy), radius + random.randint(2, 4), 1.0, -1)
            halo_mask = cv2.GaussianBlur(halo_mask, (7, 7), 1.5)
            img[:, :, 0] = np.clip(img[:, :, 0] + halo_mask * 0.35, 0.0, 1.0)
            img[:, :, 1] = np.clip(img[:, :, 1] + halo_mask * 0.25, 0.0, 1.0)
            img[:, :, 2] = np.clip(img[:, :, 2] - halo_mask * 0.20, 0.0, 1.0)

            # Necrotic brown/dark core
            core_mask = np.zeros((th, tw), dtype=np.float32)
            cv2.circle(core_mask, (cx, cy), radius, 1.0, -1)
            core_mask = cv2.GaussianBlur(core_mask, (3, 3), 0.8)
            img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 - core_mask * 0.7) + core_mask * 0.25, 0.0, 1.0)
            img[:, :, 1] = np.clip(img[:, :, 1] * (1.0 - core_mask * 0.8) + core_mask * 0.15, 0.0, 1.0)
            img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 - core_mask * 0.9) + core_mask * 0.05, 0.0, 1.0)

        return img

    def _synthesize_pest_infestation(self, base_rgb: np.ndarray) -> np.ndarray:
        """Injects chewed defoliation holes, irregular bite margins, and bite stippling."""
        img = base_rgb.copy()
        th, tw, _ = img.shape
        num_holes = random.randint(4, 10)
        for _ in range(num_holes):
            cx = random.randint(6, tw - 7)
            cy = random.randint(6, th - 7)
            rx = random.randint(2, 6)
            ry = random.randint(2, 6)
            angle = random.randint(0, 180)

            # Chewed void (soil/dark ground show-through)
            hole_mask = np.zeros((th, tw), dtype=np.float32)
            cv2.ellipse(hole_mask, (cx, cy), (rx, ry), angle, 0, 360, 1.0, -1)
            hole_mask = cv2.GaussianBlur(hole_mask, (3, 3), 0.5)

            # Soil / shadow color under foliage
            soil_color = np.array([0.15, 0.12, 0.08], dtype=np.float32)
            for c in range(3):
                img[:, :, c] = img[:, :, c] * (1.0 - hole_mask) + soil_color[c] * hole_mask

        # Add stippling bite puncturing
        num_stipples = random.randint(10, 25)
        for _ in range(num_stipples):
            sx = random.randint(2, tw - 3)
            sy = random.randint(2, th - 3)
            img[sy, sx, :] = [0.05, 0.05, 0.05]

        return img

    def _synthesize_water_stress(self, base_rgb: np.ndarray) -> np.ndarray:
        """Injects diffuse drought chlorosis, desiccation yellowing/browning, and contrast drop."""
        img = base_rgb.copy()
        th, tw, _ = img.shape

        # Smooth drought gradient across the patch
        grad = np.linspace(0.4, 0.9, tw, dtype=np.float32)[None, :]
        if random.random() > 0.5:
            grad = np.fliplr(grad)

        # Shift green hue toward dry yellow/straw: increase Red, decrease Green & Blue
        img[:, :, 0] = np.clip(img[:, :, 0] + grad * 0.30, 0.0, 1.0)
        img[:, :, 1] = np.clip(img[:, :, 1] * (1.0 - grad * 0.25) + grad * 0.25, 0.0, 1.0)
        img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 - grad * 0.40), 0.0, 1.0)

        # Desiccation contrast reduction
        img = (img - 0.5) * 0.8 + 0.5
        return np.clip(img, 0.0, 1.0)

    def _synthesize_weed_invasion(self, base_rgb: np.ndarray) -> np.ndarray:
        """Injects dense foreign vegetative clumping (broadleaf / distinct texture)."""
        img = base_rgb.copy()
        th, tw, _ = img.shape
        cx = random.randint(16, tw - 17)
        cy = random.randint(16, th - 17)

        # Broadleaf weed cluster mask
        weed_mask = np.zeros((th, tw), dtype=np.float32)
        for _ in range(random.randint(3, 6)):
            ox = cx + random.randint(-8, 8)
            oy = cy + random.randint(-8, 8)
            cv2.circle(weed_mask, (ox, oy), random.randint(6, 12), 1.0, -1)
        weed_mask = cv2.GaussianBlur(weed_mask, (5, 5), 1.2)

        # Weed spectral signature: vibrant lime-green or dark broadleaf foliage
        weed_r = random.uniform(0.15, 0.35)
        weed_g = random.uniform(0.70, 0.95)
        weed_b = random.uniform(0.10, 0.25)

        img[:, :, 0] = img[:, :, 0] * (1.0 - weed_mask) + weed_r * weed_mask
        img[:, :, 1] = img[:, :, 1] * (1.0 - weed_mask) + weed_g * weed_mask
        img[:, :, 2] = img[:, :, 2] * (1.0 - weed_mask) + weed_b * weed_mask

        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.external_images:
            path, class_idx = self.external_images[idx]
            bgr = cv2.imread(path)
            if bgr is None:
                rgb = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (self.target_size[1], self.target_size[0])).astype(np.float32) / 255.0
        else:
            class_idx = idx % self.num_classes
            base_rgb = self._load_base_patch(idx)

            if class_idx == 0:
                # Healthy Canopy
                rgb = base_rgb
            elif class_idx == 1:
                # Fungal/Bacterial Disease
                rgb = self._synthesize_fungal_disease(base_rgb)
            elif class_idx == 2:
                # Pest Infestation
                rgb = self._synthesize_pest_infestation(base_rgb)
            elif class_idx == 3:
                # Water Stress
                rgb = self._synthesize_water_stress(base_rgb)
            elif class_idx == 4:
                # Weed Invasion
                rgb = self._synthesize_weed_invasion(base_rgb)
            else:
                rgb = base_rgb

        # Augmentations for training
        if self.is_train:
            if random.random() > 0.5:
                rgb = np.fliplr(rgb)
            if random.random() > 0.5:
                rgb = np.flipud(rgb)
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                rgb = np.rot90(rgb, k)
            # Mild brightness jitter
            jitter = random.uniform(-0.05, 0.05)
            rgb = np.clip(rgb + jitter, 0.0, 1.0)

        # Convert HWC [0, 1] to CHW PyTorch tensor
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()
        return tensor, class_idx
