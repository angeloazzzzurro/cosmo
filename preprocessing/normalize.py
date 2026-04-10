"""
COSMO — Normalizzazione curve di luce
"""

import numpy as np


def median_normalize(flux: np.ndarray, kernel_size: int = 201) -> np.ndarray:
    """
    Rimuove la variabilità stellare lenta con un filtro mediano,
    poi normalizza: flux_norm = flux / trend - 1  (transiti → negativi)
    """
    from scipy.ndimage import median_filter
    trend = median_filter(flux, size=kernel_size, mode="reflect")
    trend = np.where(trend == 0, 1e-8, trend)
    normalized = flux / trend - 1.0
    return normalized.astype(np.float32)


def clip_normalize(view: np.ndarray, clip_sigma: float = 5.0) -> np.ndarray:
    """Clipping outlier + rescaling in [-1, 1]."""
    mu, sigma = np.median(view), view.std()
    view = np.clip(view, mu - clip_sigma * sigma, mu + clip_sigma * sigma)
    vmin, vmax = view.min(), view.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(view)
    return 2.0 * (view - vmin) / (vmax - vmin) - 1.0
