"""
difference_engine.py
====================
Seam and adapters for computing spectral difference heatmaps between restored
canopy frames and baseline autoencoder reconstructions.
"""

from abc import ABC, abstractmethod
import numpy as np
import cv2

try:
    import agri_core
    HAS_AGRI_CORE = True
except ImportError:
    HAS_AGRI_CORE = False


class DifferenceEngine(ABC):
    """
    Interface for computing pixel-wise spectral difference heatmaps.
    Callers and tests cross this seam regardless of whether hardware-accelerated
    C++ or pure Python/OpenCV is active.
    """

    @abstractmethod
    def compute_heatmap(
        self, clean_frame: np.ndarray, recon_frame: np.ndarray, sigma: float = 1.5
    ) -> np.ndarray:
        """
        Computes pixel-wise squared error smoothed by a Gaussian spatial filter.

        Args:
            clean_frame: Clean 2D array [H, W] float32 in [0, 1].
            recon_frame: Reconstructed baseline 2D array [H, W] float32 in [0, 1].
            sigma: Gaussian smoothing standard deviation.

        Returns:
            heatmap: 2D array [H, W] float32 smoothed MSE matrix.
        """
        pass


class OpenCVDifferenceAdapter(DifferenceEngine):
    """
    Fast, portable adapter using NumPy vectorization and OpenCV Gaussian blur.
    Always available across development and testing environments.
    """

    def compute_heatmap(
        self, clean_frame: np.ndarray, recon_frame: np.ndarray, sigma: float = 1.5
    ) -> np.ndarray:
        diff = clean_frame.astype(np.float32) - recon_frame.astype(np.float32)
        mse = diff * diff
        return cv2.GaussianBlur(mse, (0, 0), sigmaX=sigma, sigmaY=sigma)


class NativeDifferenceAdapter(DifferenceEngine):
    """
    Hardware-accelerated adapter leveraging native C++ agri_core extension.
    Falls back to OpenCV adapter if the C++ binding encounters runtime errors.
    """

    def __init__(self, mse_threshold: float = 0.06):
        self.mse_threshold = mse_threshold
        self._fallback = OpenCVDifferenceAdapter()

    def compute_heatmap(
        self, clean_frame: np.ndarray, recon_frame: np.ndarray, sigma: float = 1.5
    ) -> np.ndarray:
        if not HAS_AGRI_CORE:
            return self._fallback.compute_heatmap(clean_frame, recon_frame, sigma)

        try:
            calc = agri_core.AnomalyCalculus(self.mse_threshold, sigma)
            if hasattr(calc, "computePixelwiseMSE"):
                c_mat = (clean_frame * 255.0).astype(np.uint8)
                r_mat = (recon_frame * 255.0).astype(np.uint8)
                heatmap = np.zeros_like(clean_frame, dtype=np.float32)
                calc.computePixelwiseMSE(c_mat, r_mat, heatmap)
                return heatmap
        except Exception:
            pass

        return self._fallback.compute_heatmap(clean_frame, recon_frame, sigma)


def create_difference_engine(prefer_native: bool = True) -> DifferenceEngine:
    """Factory creating the appropriate difference engine adapter."""
    if prefer_native and HAS_AGRI_CORE:
        return NativeDifferenceAdapter()
    return OpenCVDifferenceAdapter()
