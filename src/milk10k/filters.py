"""Spatial filters from Session 4: kernels, Gaussian blur, Sobel and a from-scratch Canny.

All functions take a 2-D float array (grayscale) and return arrays of the same shape.
Convolution here is *correlation* (the kernel is not flipped), which is what both
``scipy.ndimage.correlate`` and ``torch.nn.Conv2d`` compute - so a kernel written
here can be dropped into a Conv2d unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

SOBEL_X = np.outer([1, 2, 1], [-1, 0, 1]).astype(float)   # == ndimage.sobel(axis=1)
SOBEL_Y = SOBEL_X.T                                        # == ndimage.sobel(axis=0)
PREWITT_X = np.outer([1, 1, 1], [-1, 0, 1]).astype(float)
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)
BOX_3 = np.full((3, 3), 1 / 9)


def gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """1-D Gaussian weights as ``scipy.ndimage.gaussian_filter`` builds them:
    radius = int(truncate * sigma + 0.5), so the window is ~ +-4 sigma wide."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    return w / w.sum()


def gabor_kernel(theta: float, wavelength: float = 8.0, sigma: float = 4.0,
                 size: int = 21) -> np.ndarray:
    """Real Gabor: a sinusoid of the given wavelength under a Gaussian window,
    oriented so it responds to edges/stripes whose normal points along ``theta``."""
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    xr = x * np.cos(theta) + y * np.sin(theta)
    envelope = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    k = envelope * np.cos(2 * np.pi * xr / wavelength)
    return k - k.mean()                      # zero DC: flat skin gives zero response


def dog_kernel(sigma_small: float = 1.0, sigma_big: float = 2.0, size: int = 15) -> np.ndarray:
    """Difference of Gaussians: a band-pass blob detector (centre minus surround)."""
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    r2 = x**2 + y**2
    g = lambda s: np.exp(-r2 / (2 * s**2)) / (2 * np.pi * s**2)
    return g(sigma_small) - g(sigma_big)


def kernel_bank() -> dict[str, np.ndarray]:
    """The hand-designed filters used in the Session 4 notebook."""
    return {
        "Sobel x": SOBEL_X, "Sobel y": SOBEL_Y, "Prewitt x": PREWITT_X,
        "Laplacian": LAPLACIAN, "Sharpen": SHARPEN,
        "Gabor 0°": gabor_kernel(0.0), "Gabor 45°": gabor_kernel(np.pi / 4),
        "Gabor 90°": gabor_kernel(np.pi / 2), "DoG": dog_kernel(),
    }


def apply_kernel(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return ndimage.correlate(gray.astype(float), kernel, mode="reflect")


# --------------------------------------------------------------------------
# Gradients and edge density
# --------------------------------------------------------------------------

def sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(gx, gy, magnitude): change left->right, top->bottom, and hypot of the two."""
    gray = gray.astype(float)
    gx, gy = ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0)
    return gx, gy, np.hypot(gx, gy)


def edge_density(gray: np.ndarray) -> float:
    """Mean Sobel magnitude - the Session 5 'edge density' feature."""
    return float(sobel_gradients(gray)[2].mean())


def sobel_edges(gray: np.ndarray, threshold: float = 0.2, sigma: float = 0.0) -> np.ndarray:
    """Plain Sobel edge map: optional blur, then magnitude >= threshold * max."""
    if sigma > 0:
        gray = ndimage.gaussian_filter(gray.astype(float), sigma)
    mag = sobel_gradients(gray)[2]
    return mag >= threshold * mag.max()


# --------------------------------------------------------------------------
# Canny
# --------------------------------------------------------------------------

def non_max_suppression(magnitude: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Keep a pixel only if it is the largest of its two neighbours ALONG the gradient
    direction (quantised to 0/45/90/135 degrees); everything else becomes 0. This is
    what turns Sobel's thick ridges into 1-pixel-wide lines."""
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0          # image coords: y points down
    sector = ((angle + 22.5) // 45).astype(int) % 4          # 0:0°  1:45°  2:90°  3:135°
    # neighbour offsets (dy, dx) on each side of the pixel, per sector
    offsets = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}
    padded = np.pad(magnitude, 1, mode="edge")
    h, w = magnitude.shape
    keep = np.zeros(magnitude.shape, dtype=bool)
    for s, (dy, dx) in offsets.items():
        ahead = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        behind = padded[1 - dy:1 - dy + h, 1 - dx:1 - dx + w]
        keep |= (sector == s) & (magnitude >= ahead) & (magnitude >= behind)
    return np.where(keep, magnitude, 0.0)


def hysteresis(thin: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep pixels >= ``high`` (strong) plus every pixel >= ``low`` (weak) that is
    8-connected to a strong one. Weak pixels with no strong anchor are noise."""
    strong = thin >= high
    weak = thin >= low
    labels, n = ndimage.label(weak, structure=np.ones((3, 3), dtype=int))
    anchored = np.bincount(labels[strong], minlength=n + 1) > 0
    anchored[0] = False                                      # label 0 = background
    return anchored[labels]


def canny(gray: np.ndarray, sigma: float = 1.5, low: float = 0.05,
          high: float = 0.15) -> np.ndarray:
    """Canny edge detector -> boolean edge map.

    1. Gaussian blur (``sigma``) to suppress noise and fine texture.
    2. Sobel gradient magnitude and direction.
    3. Non-maximum suppression: thin ridges to one pixel.
    4. Double threshold + hysteresis. ``low`` and ``high`` are fractions of the
       strongest edge in the image, so the result does not depend on exposure.
    """
    smoothed = ndimage.gaussian_filter(gray.astype(float), sigma)
    gx, gy, mag = sobel_gradients(smoothed)
    thin = non_max_suppression(mag, gx, gy)
    peak = thin.max()
    if peak == 0:
        return np.zeros(gray.shape, dtype=bool)
    return hysteresis(thin, low * peak, high * peak)
