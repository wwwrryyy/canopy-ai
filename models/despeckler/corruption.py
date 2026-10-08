"""
corruption.py
=============
Aerial drone sensor degradation pipeline.

Replaces SAR multiplicative noise with three degradation types characteristic
of drone-mounted line-scan and frame cameras:

  1. Atmospheric haze     — optical depth attenuation + additive airlight.
  2. Motion blur          — linear blur kernel from drone platform vibration.
  3. Gaussian read noise  — additive sensor amplifier noise.

For Sentinel-1 SAR inputs the original Rayleigh/Gamma multiplicative model
is retained and selectable via the 'sar' mode.

Public interface (unchanged so train.py requires no edits):
  - corrupt_image(image, rng=None) -> np.ndarray
  - generate_pair(clean, rng=None) -> (corrupted, clean)
  - load_and_normalize(path, grayscale=True) -> np.ndarray
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Individual degradation primitives
# ---------------------------------------------------------------------------

def apply_atmospheric_haze(
    image: np.ndarray,
    transmission: float = 0.6,
    airlight: float = 0.9,
) -> np.ndarray:
    """Simulate atmospheric haze via Koschmieder's model: I = t*J + (1-t)*A.

    Args:
        image:        Clean image in [0, 1] float32, shape (H, W) or (H, W, C).
        transmission: Fraction of scene radiance that survives (lower = hazier).
                      Typical drone range: 0.4 – 0.85.
        airlight:     Atmospheric light intensity in [0, 1]. Near-white (0.85–1.0)
                      is physically realistic for haze/fog conditions.

    Returns:
        Hazy image clipped to [0, 1] float32.
    """
    hazy = transmission * image + (1.0 - transmission) * airlight
    return np.clip(hazy, 0.0, 1.0).astype(np.float32)


def apply_motion_blur(
    image: np.ndarray,
    kernel_size: int = 9,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Simulate linear motion blur from drone vibration or forward flight.

    Args:
        image:       Clean image in [0, 1] float32, shape (H, W) or (H, W, C).
        kernel_size: Length of the blur kernel in pixels (must be odd, ≥ 3).
                     Typical drone vibration: 5–15 px.
        angle_deg:   Blur direction in degrees (0 = horizontal).

    Returns:
        Blurred image clipped to [0, 1] float32.
    """
    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Build a line kernel and rotate it to the requested angle
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel_size

    # Rotate via warpAffine so we can blur at arbitrary angles
    center = (kernel_size // 2, kernel_size // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (kernel_size, kernel_size))
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum

    if image.ndim == 2:
        blurred = cv2.filter2D(image, -1, kernel)
    else:
        blurred = cv2.filter2D(image, -1, kernel)

    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def apply_gaussian_read_noise(
    image: np.ndarray,
    sigma: float = 0.04,
) -> np.ndarray:
    """Simulate additive Gaussian sensor read noise (CCD/CMOS amplifier thermal noise).

    Args:
        image: Clean image in [0, 1] float32, shape (H, W) or (H, W, C).
        sigma: Standard deviation of the noise. Typical sensor range: 0.01–0.08.

    Returns:
        Noisy image clipped to [0, 1] float32.
    """
    noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0).astype(np.float32)


def apply_rayleigh_speckle(image: np.ndarray, scale: float = 0.3) -> np.ndarray:
    """Apply multiplicative Rayleigh speckle (Sentinel-1 SAR single-look).

    Retained for SAR input paths. Not used in the default drone pipeline.

    Args:
        image: Clean image in [0, 1] float32, shape (H, W) or (H, W, C).
        scale: Rayleigh scale parameter.

    Returns:
        Corrupted image clipped to [0, 1] float32.
    """
    noise = np.random.rayleigh(scale=scale, size=image.shape).astype(np.float32)
    noise = noise / (scale * np.sqrt(np.pi / 2))
    return np.clip(image * noise, 0.0, 1.0).astype(np.float32)


def apply_gamma_speckle(image: np.ndarray, num_looks: float = 3.0) -> np.ndarray:
    """Apply multiplicative Gamma speckle (Sentinel-1 SAR multi-look).

    Retained for SAR input paths. Not used in the default drone pipeline.

    Args:
        image:     Clean image in [0, 1] float32, shape (H, W) or (H, W, C).
        num_looks: Number of looks (shape parameter). Lower = noisier.

    Returns:
        Corrupted image clipped to [0, 1] float32.
    """
    noise = np.random.gamma(
        shape=num_looks, scale=1.0 / num_looks, size=image.shape
    ).astype(np.float32)
    return np.clip(image * noise, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Composite corruption entry point
# ---------------------------------------------------------------------------

def corrupt_image(image: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """Apply randomised aerial drone degradation to a clean image.

    Randomly samples one of four degradation modes:
      - 'haze'        : atmospheric haze only
      - 'blur'        : motion blur only
      - 'noise'       : Gaussian read noise only
      - 'combined'    : all three chained (haze → blur → noise)
      - 'sar'         : Rayleigh + Gamma multiplicative speckle (Sentinel-1 path)

    The 'combined' mode is sampled at 2× probability to produce diverse, hard
    training pairs that exercise all denoising capabilities.

    Args:
        image: Clean image in [0, 1] float32, shape (H, W) or (H, W, C).
        rng:   Optional numpy Generator for reproducibility.

    Returns:
        Corrupted image in [0, 1] float32, same shape as input.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Weighted mode selection: combined appears twice to raise its probability
    mode = rng.choice(
        ["haze", "blur", "noise", "combined", "combined", "sar"],
        p=[0.15, 0.15, 0.15, 0.30, 0.15, 0.10],
    )

    if mode == "haze":
        t = float(rng.uniform(0.4, 0.85))
        a = float(rng.uniform(0.80, 1.00))
        return apply_atmospheric_haze(image, transmission=t, airlight=a)

    elif mode == "blur":
        k = int(rng.integers(5, 16, endpoint=True))
        angle = float(rng.uniform(0.0, 180.0))
        return apply_motion_blur(image, kernel_size=k, angle_deg=angle)

    elif mode == "noise":
        sigma = float(rng.uniform(0.01, 0.08))
        return apply_gaussian_read_noise(image, sigma=sigma)

    elif mode == "sar":
        # Sentinel-1 SAR path — chain Rayleigh + Gamma
        scale = float(rng.uniform(0.15, 0.45))
        num_looks = float(rng.uniform(1.5, 5.0))
        result = apply_rayleigh_speckle(image, scale=scale)
        return apply_gamma_speckle(result, num_looks=num_looks)

    else:  # combined: haze → blur → noise
        t = float(rng.uniform(0.45, 0.85))
        a = float(rng.uniform(0.80, 1.00))
        k = int(rng.integers(3, 13, endpoint=True))
        angle = float(rng.uniform(0.0, 180.0))
        sigma = float(rng.uniform(0.01, 0.06))
        result = apply_atmospheric_haze(image, transmission=t, airlight=a)
        result = apply_motion_blur(result, kernel_size=k, angle_deg=angle)
        result = apply_gaussian_read_noise(result, sigma=sigma)
        return result


# ---------------------------------------------------------------------------
# Unchanged helpers (train.py depends on these signatures)
# ---------------------------------------------------------------------------

def load_and_normalize(path: str, grayscale: bool = True) -> np.ndarray:
    """Load an image from disk and normalize to [0, 1] float32.

    Args:
        path:      Path to image file.
        grayscale: If True, load as single-channel grayscale.

    Returns:
        Normalized image array, shape (H, W) or (H, W, C).
    """
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img.astype(np.float32) / 255.0


def generate_pair(
    clean: np.ndarray, rng: np.random.Generator = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a (corrupted, clean) training pair from a clean image.

    Args:
        clean: Clean image in [0, 1] float32.
        rng:   Optional numpy Generator.

    Returns:
        Tuple of (corrupted, clean) both in [0, 1] float32.
    """
    corrupted = corrupt_image(clean, rng=rng)
    return corrupted, clean
