"""
difference_engine.py
====================
Seam and adapters for computing spectral difference heatmaps between restored
canopy frames and baseline autoencoder reconstructions.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

try:
    import agri_core
    HAS_AGRI_CORE = True
except ImportError:
    HAS_AGRI_CORE = False


@dataclass
class DifferenceConfig:
    """
    Hyperparameters and presets for adaptive structural difference calculation.
    """
    w_dssim: float = 0.5
    w_lncc: float = 0.3
    w_mse: float = 0.2
    window_size: int = 11
    sigma: float = 1.5
    c1: float = 0.0001   # (0.01)^2 stability constant for luminance
    c2: float = 0.0009   # (0.03)^2 stability constant for contrast
    eps: float = 1e-4

    def __post_init__(self):
        if self.window_size <= 0 or self.window_size % 2 == 0:
            raise ValueError(f"window_size must be a positive odd integer, got {self.window_size}")

    @classmethod
    def balanced(cls) -> "DifferenceConfig":
        """Balanced preset for general agricultural flight conditions."""
        return cls(w_dssim=0.5, w_lncc=0.3, w_mse=0.2, window_size=11, sigma=1.5)

    @classmethod
    def cloud_shadow_resilient(cls) -> "DifferenceConfig":
        """High-structural weighting suppressing passing cloud shadows and uneven lighting."""
        return cls(w_dssim=0.6, w_lncc=0.3, w_mse=0.1, window_size=11, sigma=1.5)

    @classmethod
    def high_sensitivity(cls) -> "DifferenceConfig":
        """Higher spectral error weighting for subtle early-stage foliar chlorosis."""
        return cls(w_dssim=0.4, w_lncc=0.3, w_mse=0.3, window_size=11, sigma=1.5)


class DifferenceEngine(ABC):
    """
    Interface for computing pixel-wise spectral difference heatmaps.
    Callers and tests cross this seam regardless of whether hardware-accelerated
    C++ or pure Python/OpenCV is active.
    """

    @abstractmethod
    def compute_heatmap(
        self,
        clean_frame: np.ndarray,
        recon_frame: np.ndarray,
        sigma: Optional[float] = None,
        config: Optional[DifferenceConfig] = None,
    ) -> np.ndarray:
        """
        Computes adaptive structural difference heatmap between clean and reconstructed canopy.

        Args:
            clean_frame: Clean 2D array [H, W] float32 in [0, 1].
            recon_frame: Reconstructed baseline 2D array [H, W] float32 in [0, 1].
            sigma: Optional spatial smoothing standard deviation. Defaults to config.sigma if None.
            config: Optional DifferenceConfig tuning weights and window parameters.

        Returns:
            heatmap: 2D array [H, W] float32 in [0, 1].
        """
        pass


class OpenCVDifferenceAdapter(DifferenceEngine):
    """
    Fast, portable adapter using NumPy vectorization and OpenCV Gaussian filters.
    Computes an adaptive structural composite: w_dssim * DSSIM + w_lncc * (1-LNCC) + w_mse * NormMSE.
    """

    def __init__(self, config: Optional[DifferenceConfig] = None):
        self.config = config or DifferenceConfig.balanced()

    def compute_heatmap(
        self,
        clean_frame: np.ndarray,
        recon_frame: np.ndarray,
        sigma: Optional[float] = None,
        config: Optional[DifferenceConfig] = None,
    ) -> np.ndarray:
        cfg = config or self.config
        eff_sigma = sigma if sigma is not None else cfg.sigma

        clean = clean_frame.astype(np.float32)
        recon = recon_frame.astype(np.float32)

        ksize = (cfg.window_size, cfg.window_size)
        mu_x = cv2.GaussianBlur(clean, ksize, eff_sigma)
        mu_y = cv2.GaussianBlur(recon, ksize, eff_sigma)
        mu_xx = mu_x * mu_x
        mu_yy = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_xx = np.maximum(cv2.GaussianBlur(clean * clean, ksize, eff_sigma) - mu_xx, 0.0)
        sigma_yy = np.maximum(cv2.GaussianBlur(recon * recon, ksize, eff_sigma) - mu_yy, 0.0)
        sigma_xy = cv2.GaussianBlur(clean * recon, ksize, eff_sigma) - mu_xy

        # 1. Structural Dissimilarity (DSSIM)
        num = (2.0 * mu_xy + cfg.c1) * (2.0 * sigma_xy + cfg.c2)
        den = (mu_xx + mu_yy + cfg.c1) * (sigma_xx + sigma_yy + cfg.c2)
        ssim = np.clip(num / den, -1.0, 1.0)
        dssim = (1.0 - ssim) * 0.5

        # 2. Stabilized Local Normalized Cross-Correlation Distance (1 - LNCC)
        lncc = np.clip((2.0 * sigma_xy + cfg.c2) / (sigma_xx + sigma_yy + cfg.c2), -1.0, 1.0)
        dist_lncc = (1.0 - lncc) * 0.5

        # 3. Local MSE
        diff = clean - recon
        mse_local = cv2.GaussianBlur(diff * diff, ksize, eff_sigma)

        # Composite Heatmap
        heatmap = cfg.w_dssim * dssim + cfg.w_lncc * dist_lncc + cfg.w_mse * mse_local
        return np.clip(heatmap, 0.0, 1.0).astype(np.float32)


class NativeDifferenceAdapter(DifferenceEngine):
    """
    Hardware-accelerated adapter leveraging native C++ agri_core extension.
    Falls back to OpenCV adapter if the C++ binding encounters runtime errors.
    """

    def __init__(
        self,
        mse_threshold: float = 0.06,
        config: Optional[DifferenceConfig] = None,
    ):
        self.mse_threshold = mse_threshold
        self.config = config or DifferenceConfig.balanced()
        self._fallback = OpenCVDifferenceAdapter(config=self.config)

    def compute_heatmap(
        self,
        clean_frame: np.ndarray,
        recon_frame: np.ndarray,
        sigma: Optional[float] = None,
        config: Optional[DifferenceConfig] = None,
    ) -> np.ndarray:
        cfg = config or self.config
        eff_sigma = sigma if sigma is not None else cfg.sigma

        if not HAS_AGRI_CORE:
            return self._fallback.compute_heatmap(clean_frame, recon_frame, eff_sigma, cfg)

        try:
            calc = agri_core.AnomalyCalculus(self.mse_threshold, eff_sigma)
            if hasattr(calc, "compute_structural_anomaly"):
                c_mat = clean_frame.astype(np.float32)
                r_mat = recon_frame.astype(np.float32)
                heatmap, _ = calc.compute_structural_anomaly(
                    c_mat,
                    r_mat,
                    w_dssim=cfg.w_dssim,
                    w_lncc=cfg.w_lncc,
                    w_mse=cfg.w_mse,
                    window_size=cfg.window_size,
                    sigma=eff_sigma,
                    c1=cfg.c1,
                    c2=cfg.c2,
                )
                return heatmap.copy()
        except ValueError:
            raise
        except Exception as e:
            logger.warning("Native structural anomaly calculation failed; falling back to OpenCV: %s", e)

        return self._fallback.compute_heatmap(clean_frame, recon_frame, eff_sigma, cfg)


def create_difference_engine(
    prefer_native: bool = True,
    config: Optional[DifferenceConfig] = None,
) -> DifferenceEngine:
    """Factory creating the appropriate difference engine adapter."""
    if prefer_native and HAS_AGRI_CORE:
        return NativeDifferenceAdapter(config=config)
    return OpenCVDifferenceAdapter(config=config)
