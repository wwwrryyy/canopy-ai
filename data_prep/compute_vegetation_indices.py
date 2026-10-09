"""
compute_vegetation_indices.py
==============================
Vegetation index computation for multispectral drone imagery.

Computes NDVI and NDRE from raw multispectral band arrays and appends
them as additional channels to the image tensor fed into Stage 1 (DespecklerUNet).

Supported inputs
----------------
- 5-band multispectral stack [B, G, R, RedEdge, NIR] in [0, 1] float32
- 3-band RGB-only (NDVI/NDRE computed as stubs returning zeros when bands
  are absent, so downstream code handles missing sensors gracefully)

Output
------
A 7-channel numpy array [B, G, R, RedEdge, NIR, NDVI, NDRE] in [0, 1] float32.
NDVI and NDRE are normalized from [-1, 1] → [0, 1] before stacking.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Index primitives
# ---------------------------------------------------------------------------

def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red + ε)

    Healthy vegetation: 0.6 – 0.9
    Bare soil / stressed: 0.0 – 0.3
    Water / non-vegetated: < 0.0

    Args:
        nir: NIR band in [0, 1] float32, shape (H, W).
        red: Red band in [0, 1] float32, shape (H, W).

    Returns:
        NDVI in [-1, 1] float32, shape (H, W).
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return (nir - red) / (nir + red + 1e-8)


def compute_ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    """Normalized Difference Red Edge index.

    NDRE = (NIR - RedEdge) / (NIR + RedEdge + ε)

    More sensitive than NDVI to early-stage crop stress (chlorophyll changes
    in upper canopy layers before full senescence).

    Args:
        nir:       NIR band in [0, 1] float32, shape (H, W).
        red_edge:  RedEdge band in [0, 1] float32, shape (H, W).

    Returns:
        NDRE in [-1, 1] float32, shape (H, W).
    """
    nir = nir.astype(np.float32)
    red_edge = red_edge.astype(np.float32)
    return (nir - red_edge) / (nir + red_edge + 1e-8)


def normalize_index(index: np.ndarray) -> np.ndarray:
    """Map a vegetation index from [-1, 1] to [0, 1].

    Required so index channels are on the same scale as the raw bands
    before feeding into the DespecklerUNet encoder.

    Args:
        index: Index array in [-1, 1] float32.

    Returns:
        Normalized array in [0, 1] float32.
    """
    return np.clip((index + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main composition function
# ---------------------------------------------------------------------------

def append_vegetation_indices(bands: np.ndarray) -> np.ndarray:
    """Append NDVI and NDRE channels to a multispectral band stack.

    Args:
        bands: Multispectral array in [0, 1] float32.
               Accepted shapes:
               - (H, W, 5)  — [B, G, R, RedEdge, NIR]
               - (H, W, 3)  — [B, G, R]  (NDVI/NDRE set to 0.5 i.e. index=0)
               - (H, W)     — single grayscale (indices set to 0.5)

    Returns:
        Array of shape:
        - (H, W, 7)  for 5-band input  — [B, G, R, RedEdge, NIR, NDVI, NDRE]
        - (H, W, 5)  for 3-band input  — [B, G, R, NDVI_stub, NDRE_stub]
        - (H, W, 3)  for 1-band input  — [Gray, NDVI_stub, NDRE_stub]
        All channels remain in [0, 1] float32.

    Raises:
        ValueError: If the input has an unsupported number of channels.
    """
    if bands.ndim == 2:
        # Grayscale — no real spectral bands; append neutral stubs
        h, w = bands.shape
        stub = np.full((h, w), 0.5, dtype=np.float32)
        return np.stack([bands.astype(np.float32), stub, stub], axis=-1)

    h, w, c = bands.shape

    if c == 5:
        # Full 5-band multispectral: [B, G, R, RedEdge, NIR]
        red = bands[:, :, 2]
        red_edge = bands[:, :, 3]
        nir = bands[:, :, 4]

        ndvi = normalize_index(compute_ndvi(nir, red))
        ndre = normalize_index(compute_ndre(nir, red_edge))

        return np.concatenate(
            [bands.astype(np.float32), ndvi[:, :, np.newaxis], ndre[:, :, np.newaxis]],
            axis=-1,
        )

    elif c == 3:
        # RGB only — no NIR/RedEdge available; stubs at 0.5 (neutral index)
        h, w = bands.shape[:2]
        stub = np.full((h, w, 1), 0.5, dtype=np.float32)
        return np.concatenate([bands.astype(np.float32), stub, stub], axis=-1)

    else:
        raise ValueError(
            f"Expected 1, 3, or 5 channels, got {c}. "
            "Provide bands in [B, G, R] or [B, G, R, RedEdge, NIR] order."
        )


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_multispectral(paths: dict) -> np.ndarray:
    """Load individual band images from disk and stack into a (H, W, N) array.

    Args:
        paths: Dict mapping band name to file path. Recognised keys:
               'blue', 'green', 'red', 'red_edge', 'nir'.
               At minimum 'red' and 'nir' are needed for real NDVI.

    Returns:
        Band stack in [0, 1] float32, shape (H, W, N) where N = len(paths),
        ordered [blue, green, red, red_edge, nir] for the keys present.

    Raises:
        FileNotFoundError: If any specified path cannot be read.
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
