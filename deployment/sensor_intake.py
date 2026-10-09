"""
sensor_intake.py
================
Deep Sensor Intake module for drone-mounted multispectral cameras, SAR feeds, and RGB sensors.

Ingests variable sensor payloads, normalizes floating-point radiance, dynamically synthesizes
vegetation indices (NDVI, NDRE), and delivers calibrated MultispectralCanopyFrame domain objects
with projection adapters for downstream neural models.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index: NDVI = (NIR - Red) / (NIR + Red + ε)"""
    nir_f = nir.astype(np.float32)
    red_f = red.astype(np.float32)
    return (nir_f - red_f) / (nir_f + red_f + 1e-8)


def compute_ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    """Normalized Difference Red Edge: NDRE = (NIR - RedEdge) / (NIR + RedEdge + ε)"""
    nir_f = nir.astype(np.float32)
    re_f = red_edge.astype(np.float32)
    return (nir_f - re_f) / (nir_f + re_f + 1e-8)


def normalize_index(index: np.ndarray) -> np.ndarray:
    """Maps index from [-1.0, 1.0] to [0.0, 1.0] for neural network inputs."""
    return np.clip((index + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


@dataclass
class MultispectralCanopyFrame:
    """
    Calibrated multi-band sensor capture encapsulating raw spectral bands alongside
    synthesized vegetation indices (NDVI and NDRE).
    """
    bands: Dict[str, np.ndarray]
    ndvi: Optional[np.ndarray] = None
    ndre: Optional[np.ndarray] = None
    height: int = 128
    width: int = 128

    @property
    def has_vegetation_indices(self) -> bool:
        return self.ndvi is not None and self.ndre is not None

    def to_single_channel(self, prefer_ndvi: bool = True) -> np.ndarray:
        """
        Projects frame to a 2D [H, W] float32 array in [0, 1].
        Suitable for single-channel models (Despeckler U-Net and ConvAutoencoder).
        """
        if "gray" in self.bands:
            return self.bands["gray"]

        if prefer_ndvi and self.ndvi is not None and "nir" in self.bands:
            return self.ndvi

        if "red" in self.bands and "green" in self.bands and "blue" in self.bands:
            # Standard radiometric luminance: Y = 0.299R + 0.587G + 0.114B
            lum = 0.299 * self.bands["red"] + 0.587 * self.bands["green"] + 0.114 * self.bands["blue"]
            return np.clip(lum, 0.0, 1.0).astype(np.float32)

        if "nir" in self.bands:
            return self.bands["nir"]

        # Fallback to the first available band
        if self.bands:
            first_band = next(iter(self.bands.values()))
            return first_band

        return np.full((self.height, self.width), 0.5, dtype=np.float32)

    def to_rgb_composite(self) -> np.ndarray:
        """
        Projects frame to a 3-channel [3, H, W] float32 array in [0, 1].
        Suitable for Stage 3 MobileNetV3 AgriculturalClassifier.
        """
        if "red" in self.bands and "green" in self.bands and "blue" in self.bands:
            return np.stack([self.bands["red"], self.bands["green"], self.bands["blue"]], axis=0)

        if "nir" in self.bands and "red" in self.bands and "green" in self.bands:
            # False-Color Infrared Composite (NIR -> Red, Red -> Green, Green -> Blue)
            return np.stack([self.bands["nir"], self.bands["red"], self.bands["green"]], axis=0)

        # Single-channel replication fallback
        single_2d = self.to_single_channel()
        return np.repeat(single_2d[np.newaxis, :, :], 3, axis=0).astype(np.float32)

    def to_multispectral_tensor(self) -> np.ndarray:
        """
        Exports all available bands and indices as a stacked [C, H, W] tensor in [0, 1].
        Standard order for 5-band: [Blue, Green, Red, RedEdge, NIR, NDVI, NDRE].
        """
        channels = []
        for key in ["blue", "green", "red", "red_edge", "nir"]:
            if key in self.bands:
                channels.append(self.bands[key])

        if self.ndvi is not None:
            channels.append(self.ndvi)
        if self.ndre is not None:
            channels.append(self.ndre)

        if not channels:
            channels = [self.to_single_channel()]

        return np.stack(channels, axis=0).astype(np.float32)


class SensorIntakeModule:
    """
    Deep module providing ingestion, calibration, and spectral synthesis across
    multispectral, standard RGB, and SAR drone payloads.
    """

    def __init__(self, target_size: Optional[Tuple[int, int]] = (128, 128)):
        self.target_size = target_size

    def ingest(
        self, payload: Union[np.ndarray, Dict[str, Union[np.ndarray, str]], MultispectralCanopyFrame]
    ) -> MultispectralCanopyFrame:
        """
        Ingests variable payload formats and returns a calibrated MultispectralCanopyFrame.

        Args:
            payload: One of:
              - MultispectralCanopyFrame instance
              - 2D numpy array [H, W] (grayscale / SAR)
              - 3D numpy array [H, W, 3] (RGB)
              - 3D numpy array [H, W, 5] ([B, G, R, RedEdge, NIR])
              - 3D numpy array [H, W, 7] ([B, G, R, RedEdge, NIR, NDVI, NDRE])
              - Dict mapping band names to arrays or image file paths
        """
        if isinstance(payload, MultispectralCanopyFrame):
            return payload

        if isinstance(payload, dict):
            return self._ingest_dict(payload)

        if isinstance(payload, np.ndarray):
            return self._ingest_array(payload)

        raise TypeError(f"Unsupported sensor payload type: {type(payload)}")

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Ensures float32 in range [0.0, 1.0]."""
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
            if np.max(img) > 1.0:
                img = img / 255.0
        return np.clip(img, 0.0, 1.0)

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Resizes to target dimensions if target_size is set."""
        if self.target_size is None:
            return img
        h, w = self.target_size
        if img.shape[0] != h or img.shape[1] != w:
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img

    def _ingest_array(self, arr: np.ndarray) -> MultispectralCanopyFrame:
        arr = self._normalize_image(arr)

        if arr.ndim == 2:
            # Grayscale / SAR single-band
            arr = self._resize(arr)
            h, w = arr.shape
            stub = np.full((h, w), 0.5, dtype=np.float32)
            return MultispectralCanopyFrame(
                bands={"gray": arr}, ndvi=stub, ndre=stub, height=h, width=w
            )

        h, w, c = arr.shape
        if c == 1:
            channel = self._resize(arr[:, :, 0])
            stub = np.full(channel.shape, 0.5, dtype=np.float32)
            return MultispectralCanopyFrame(
                bands={"gray": channel}, ndvi=stub, ndre=stub, height=channel.shape[0], width=channel.shape[1]
            )

        if c == 3:
            # RGB feed: [B, G, R]
            b = self._resize(arr[:, :, 0])
            g = self._resize(arr[:, :, 1])
            r = self._resize(arr[:, :, 2])
            stub = np.full(b.shape, 0.5, dtype=np.float32)
            return MultispectralCanopyFrame(
                bands={"blue": b, "green": g, "red": r},
                ndvi=stub,
                ndre=stub,
                height=b.shape[0],
                width=b.shape[1],
            )

        if c >= 5:
            # 5-band multispectral: [B, G, R, RedEdge, NIR]
            b = self._resize(arr[:, :, 0])
            g = self._resize(arr[:, :, 1])
            r = self._resize(arr[:, :, 2])
            re = self._resize(arr[:, :, 3])
            nir = self._resize(arr[:, :, 4])

            if c >= 7:
                # Pre-computed NDVI and NDRE
                ndvi = self._resize(arr[:, :, 5])
                ndre = self._resize(arr[:, :, 6])
            else:
                ndvi = normalize_index(compute_ndvi(nir, r))
                ndre = normalize_index(compute_ndre(nir, re))

            return MultispectralCanopyFrame(
                bands={"blue": b, "green": g, "red": r, "red_edge": re, "nir": nir},
                ndvi=ndvi,
                ndre=ndre,
                height=b.shape[0],
                width=b.shape[1],
            )

        raise ValueError(f"Unsupported channel count: {c}")

    def _ingest_dict(self, band_dict: Dict[str, Union[np.ndarray, str]]) -> MultispectralCanopyFrame:
        bands = {}
        ref_h, ref_w = self.target_size if self.target_size else (128, 128)

        for key, val in band_dict.items():
            if isinstance(val, str):
                img = cv2.imread(val, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(f"Could not load band '{key}' from path: {val}")
                img = self._normalize_image(img)
            else:
                img = self._normalize_image(val)

            img = self._resize(img)
            ref_h, ref_w = img.shape
            bands[key.lower()] = img

        ndvi = None
        ndre = None

        if "nir" in bands and "red" in bands:
            ndvi = normalize_index(compute_ndvi(bands["nir"], bands["red"]))

        if "nir" in bands and "red_edge" in bands:
            ndre = normalize_index(compute_ndre(bands["nir"], bands["red_edge"]))

        if ndvi is None:
            ndvi = np.full((ref_h, ref_w), 0.5, dtype=np.float32)
        if ndre is None:
            ndre = np.full((ref_h, ref_w), 0.5, dtype=np.float32)

        return MultispectralCanopyFrame(
            bands=bands, ndvi=ndvi, ndre=ndre, height=ref_h, width=ref_w
        )


def append_vegetation_indices(bands: np.ndarray) -> np.ndarray:
    """
    Backwards compatibility utility matching data_prep.compute_vegetation_indices signature.
    Appends NDVI and NDRE channels to a multispectral stack.
    """
    intake = SensorIntakeModule(target_size=None)
    frame = intake.ingest(bands)
    tensor = frame.to_multispectral_tensor()  # [C, H, W]
    return np.transpose(tensor, (1, 2, 0))    # [H, W, C]
