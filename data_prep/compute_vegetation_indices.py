"""
compute_vegetation_indices.py
==============================
Vegetation index computation for multispectral drone imagery.

Delegates core mathematical primitives and channel stacking to
deployment.sensor_intake for a single source of truth across offline training
and edge inference.
"""

import os
import sys
import numpy as np

# Ensure deployment directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEPLOYMENT_DIR = os.path.join(PROJECT_ROOT, "deployment")
if DEPLOYMENT_DIR not in sys.path:
    sys.path.insert(0, DEPLOYMENT_DIR)

from sensor_intake import (
    compute_ndvi,
    compute_ndre,
    normalize_index,
    append_vegetation_indices,
    SensorIntakeModule,
)


def load_multispectral(paths: dict) -> np.ndarray:
    """
    Load individual band images from disk and stack into a (H, W, N) array.
    Preserves exact backward compatibility with dataset preparation scripts.
    """
    import cv2

    ORDER = ["blue", "green", "red", "red_edge", "nir"]
    channels = []
    ref_shape = None

    for key in ORDER:
        if key not in paths:
            continue
        img = cv2.imread(paths[key], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load band '{key}': {paths[key]}")
        band = img.astype(np.float32) / 255.0
        if ref_shape is None:
            ref_shape = band.shape
        elif band.shape != ref_shape:
            import warnings
            warnings.warn(
                f"Band '{key}' shape {band.shape} differs from reference {ref_shape}; "
                "resizing to match.",
                stacklevel=2,
            )
            band = cv2.resize(band, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_AREA)
        channels.append(band)

    if not channels:
        raise ValueError("No valid band paths provided.")

    return np.stack(channels, axis=-1)


__all__ = [
    "compute_ndvi",
    "compute_ndre",
    "normalize_index",
    "append_vegetation_indices",
    "load_multispectral",
]
