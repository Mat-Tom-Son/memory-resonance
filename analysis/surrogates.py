"""
Surrogate generators for control experiments.
"""

from __future__ import annotations

import numpy as np


def psd_matched_surrogate(x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Construct a surrogate signal with identical amplitude spectrum but randomised phase.

    The input signal is demeaned before computing the FFT to remove any DC bias.
    DC and Nyquist phases are pinned to 0 to ensure the surrogate is real-valued
    and has matching statistical properties.

    Parameters
    ----------
    x:
        Real-valued time-series whose one-sided spectrum is preserved.
    rng:
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Surrogate signal with identical PSD but randomized phase.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("Input signal must be non-empty.")
    if rng is None:
        rng = np.random.default_rng()

    # Demean to remove DC bias before surrogate generation
    x_mean = np.mean(x)
    x_centered = x - x_mean

    X = np.fft.rfft(x_centered)
    magnitudes = np.abs(X)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=magnitudes.shape)

    # Pin DC and Nyquist phases to 0 for real-valued output
    phases[0] = 0.0
    if x.size % 2 == 0 and phases.size > 1:
        phases[-1] = 0.0  # Nyquist component only exists for even-length signals

    # Generate surrogate and restore mean
    surrogate = np.fft.irfft(magnitudes * np.exp(1j * phases), n=x.size)
    surrogate += x_mean

    return surrogate.astype(float, copy=False)

