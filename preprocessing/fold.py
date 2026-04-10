"""
COSMO — Phase folding
Trasforma una curva di luce lunga in un singolo "battito" del transito.

  - global_view : 2001 punti (intera orbita normalizzata)
  - local_view  :  201 punti (zoom x2 sul transito)
"""

import numpy as np


def phase_fold(time: np.ndarray, flux: np.ndarray, period: float, t0: float):
    """Calcola la fase di ogni punto della curva di luce."""
    phase = ((time - t0) % period) / period   # [0, 1)
    phase[phase > 0.5] -= 1.0                 # [-0.5, 0.5)
    order = np.argsort(phase)
    return phase[order], flux[order]


def bin_to_length(phase: np.ndarray, flux: np.ndarray, length: int):
    """Binning uniforme → vettore di lunghezza fissa."""
    bins = np.linspace(phase.min(), phase.max(), length + 1)
    binned = np.zeros(length)
    for i in range(length):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        binned[i] = flux[mask].mean() if mask.any() else 1.0
    return binned


def make_views(time: np.ndarray, flux: np.ndarray,
               period: float, t0: float, duration: float):
    """
    Restituisce (global_view, local_view) pronti per la CNN.

    global_view : 2001 punti — intera orbita
    local_view  :  201 punti — finestra ±2×durata attorno al transito
    """
    phase, flux_sorted = phase_fold(time, flux, period, t0)

    # Global view: intera fase
    global_view = bin_to_length(phase, flux_sorted, 2001)

    # Local view: ±2 durate attorno al transito (phase=0)
    half_width = 2.0 * duration / period
    mask = np.abs(phase) <= half_width
    if mask.sum() < 10:
        local_view = np.ones(201)
    else:
        local_view = bin_to_length(phase[mask], flux_sorted[mask], 201)

    return global_view.astype(np.float32), local_view.astype(np.float32)
