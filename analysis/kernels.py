"""
Kernel utilities for impedance matching between internal and external memory.

Provides builders for external (bath) kernels, estimators for internal kernels
from frequency-domain data, and an overlap functional O(τ_B) used to predict
resonance without running full time-domain simulations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
try:
    from scipy.signal.windows import tukey  # SciPy >= 1.4
except Exception:  # pragma: no cover - fallback
    from scipy.signal import tukey  # type: ignore


def _ensure_1d(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("Empty array")
    return arr


def build_Kext_OU(tau: np.ndarray, tau_B: float) -> np.ndarray:
    """
    Ornstein-Uhlenbeck (OU) memory kernel (L1-normalized, nonnegative):
        K_ext(τ; τ_B) = (1/τ_B) exp(-|τ|/τ_B)
    """
    tau = _ensure_1d(tau)
    tau_B = float(tau_B)
    if tau_B <= 0:
        raise ValueError("tau_B must be positive")
    k = np.exp(-np.abs(tau) / tau_B) / tau_B
    # L1 normalize for overlap fairness
    area = np.trapz(k, tau)
    if area > 0:
        k = k / area
    return k


def build_Kext_mixture(tau: np.ndarray, tau_list: Sequence[float], weights: Sequence[float]) -> np.ndarray:
    """
    Mixture of exponentials (causal, L1-normalized):
        K_ext(τ) = sum_i w_i (1/τ_i) exp(-|τ|/τ_i), with w_i ≥ 0, sum w_i = 1
    """
    tau = _ensure_1d(tau)
    taus = np.asarray(list(tau_list), dtype=float)
    ws = np.asarray(list(weights), dtype=float)
    if taus.size != ws.size:
        raise ValueError("tau_list and weights must have same length")
    if np.any(taus <= 0):
        raise ValueError("All tau_i must be positive")
    if np.any(ws < 0):
        raise ValueError("All weights must be nonnegative")
    if ws.sum() <= 0:
        raise ValueError("weights must sum to positive value")
    ws = ws / ws.sum()
    k = np.zeros_like(tau, dtype=float)
    for t_i, w_i in zip(taus, ws):
        k += w_i * np.exp(-np.abs(tau) / t_i) / t_i
    # L1 normalize (already normalized in ideal continuous limit; re-normalize numerically)
    area = np.trapz(k, tau)
    if area > 0:
        k = k / area
    return k


def estimate_Kint_from_H(omega: np.ndarray, H2: np.ndarray, *, window: str = "tukey", alpha: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate internal kernel K_int(τ) from |H(ω)|^2 via (windowed) inverse FT.

    Returns (tau, K_int_tau) with K_int nonnegative and L1-normalized.
    """
    omega = _ensure_1d(omega)
    H2 = _ensure_1d(H2)
    if omega.shape != H2.shape:
        raise ValueError("omega and H2 must have same shape")
    # Frequency grid spacing (assume uniform)
    dω = float(np.mean(np.diff(omega))) if omega.size > 1 else 1.0
    # Window to reduce IFT ringing
    if window == "tukey":
        w = tukey(len(omega), alpha=alpha)
    else:
        w = np.ones_like(omega)
    S = H2 * w
    # IFT to get an (even) real kernel; use rfft/irfft style by mirroring
    # Build symmetric spectrum over negative and positive frequencies
    S_pos = S
    S_neg = S_pos[::-1]
    S_full = np.concatenate([S_neg, S_pos])
    dΩ = dω
    # Time grid dual to frequency grid
    T = 2 * np.pi / max(dΩ, 1e-9)
    n = S_full.size
    tau = np.linspace(-T / 2, T / 2, n, dtype=float)
    # IFT (scaled); numpy ifft assumes 2π factors in exponent absent; scale accordingly
    K = np.real(np.fft.ifft(np.fft.ifftshift(S_full)))
    # Shift tau to center at 0 and sort
    K = np.fft.fftshift(K)
    # Clip negatives (numerical) and L1-normalize
    K = np.maximum(K, 0.0)
    area = np.trapz(K, tau)
    if area > 0:
        K = K / area
    return tau, K


def estimate_Kint_from_psd(freq_hz: np.ndarray, psd_one_sided: np.ndarray, *, window: str = "tukey", alpha: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate K_int(τ) from an empirical one-sided PSD via inverse FT (windowed).
    """
    freq_hz = _ensure_1d(freq_hz)
    psd_one = _ensure_1d(psd_one_sided)
    if freq_hz.shape != psd_one.shape:
        raise ValueError("freq_hz and psd must have same shape")
    omega = 2.0 * np.pi * freq_hz
    return estimate_Kint_from_H(omega, psd_one, window=window, alpha=alpha)


def overlap_O(
    tau_grid: np.ndarray,
    K_int_tau: np.ndarray,
    tau_B_values: Iterable[float],
    *,
    kext_builder: Callable[[np.ndarray, float], np.ndarray] = build_Kext_OU,
) -> NDArray[np.float64]:
    """
    Overlap functional O(τ_B) = ∫ K_int(τ) K_ext(τ; τ_B) dτ over tau_grid.
    Assumes both kernels are nonnegative and L1-normalized.
    """
    tau = _ensure_1d(tau_grid)
    K_int = _ensure_1d(K_int_tau)
    if tau.shape != K_int.shape:
        raise ValueError("tau_grid and K_int must have same shape")
    # L1-normalize defensively
    area = np.trapz(K_int, tau)
    if area > 0:
        K_int = K_int / area
    vals = []
    for tB in tau_B_values:
        K_ext = kext_builder(tau, float(tB))
        vals.append(float(np.trapz(K_int * K_ext, tau)))
    return np.asarray(vals, dtype=float)


def build_Kext_flat(tau: np.ndarray, _: float | None = None) -> np.ndarray:
    """Flat (uniform) kernel over provided tau grid (L1-normalized)."""
    tau = _ensure_1d(tau)
    k = np.ones_like(tau, dtype=float)
    area = np.trapz(k, tau)
    if area > 0:
        k = k / area
    return k


@dataclass
class SimplexScores:
    S: float
    C: float
    M: float

    def barycentric(self) -> tuple[float, float, float]:
        s, c, m = max(self.S, 0.0), max(self.C, 0.0), max(self.M, 0.0)
        total = s + c + m
        if total <= 0:
            return (0.0, 0.0, 0.0)
        return (s / total, c / total, m / total)
