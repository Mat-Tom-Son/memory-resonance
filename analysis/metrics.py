"""
Shared metrics for analysing slow-envelope behaviour in the warm-noise proxy.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, hilbert


def _ensure_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("Input array must be non-empty.")
    return arr


def baseband_power(
    x: np.ndarray,
    fs_hz: float,
    w3_rad_s: float,
    lp_frac: float,
) -> float:
    """
    Compute the demodulated baseband variance using I/Q mixing.

    Parameters
    ----------
    x:
        Real-valued time-series of the slow oscillator coordinate.
    fs_hz:
        Sampling rate in Hz.
    w3_rad_s:
        Carrier angular frequency of the slow oscillator (rad/s).
    lp_frac:
        Low-pass cutoff expressed as a fraction of w3 (0 < lp_frac < 1).
    """
    x = _ensure_1d(x)
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive.")
    if w3_rad_s <= 0:
        raise ValueError("w3_rad_s must be positive.")
    if not 0 < lp_frac < 1:
        raise ValueError("lp_frac must lie in (0, 1).")

    n = x.size
    t = np.arange(n, dtype=float) / fs_hz

    i = x * np.cos(w3_rad_s * t)
    q = -x * np.sin(w3_rad_s * t)

    cutoff_hz = (lp_frac * w3_rad_s) / (2.0 * np.pi)
    nyquist = fs_hz / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        raise ValueError("Low-pass cutoff must lie within (0, Nyquist).")

    sos = butter(4, cutoff_hz / nyquist, output="sos")
    i_lp = sosfiltfilt(sos, i)
    q_lp = sosfiltfilt(sos, q)

    return float(np.var(i_lp) + np.var(q_lp))


def narrowband_psd_power(
    x: np.ndarray,
    fs_hz: float,
    w3_rad_s: float,
    rel_bandwidth: float,
    nperseg: int,
    noverlap: int,
) -> float:
    """
    Integrate the Welch PSD over a narrow band surrounding w3.

    Parameters
    ----------
    x:
        Real-valued time-series.
    fs_hz:
        Sampling rate in Hz.
    w3_rad_s:
        Carrier angular frequency of the slow oscillator (rad/s).
    rel_bandwidth:
        Symmetric fractional bandwidth (e.g. 0.3 -> ±30% around w3).
    nperseg, noverlap:
        Welch parameters, ensuring consistency across experiments.
    """
    x = _ensure_1d(x)
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive.")
    if w3_rad_s <= 0:
        raise ValueError("w3_rad_s must be positive.")
    if rel_bandwidth <= 0 or rel_bandwidth >= 1:
        raise ValueError("rel_bandwidth must lie in (0, 1).")

    nperseg_eff = min(int(nperseg), x.size)
    if nperseg_eff < 2:
        return 0.0
    noverlap_eff = min(int(noverlap), nperseg_eff - 1)
    freqs, psd = welch(x, fs=fs_hz, nperseg=nperseg_eff, noverlap=noverlap_eff)
    w3_hz = w3_rad_s / (2.0 * np.pi)
    band_low = max(0.0, w3_hz * (1.0 - rel_bandwidth))
    band_high = w3_hz * (1.0 + rel_bandwidth)
    mask = (freqs >= band_low) & (freqs <= band_high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def hilbert_envelope_power(x: np.ndarray) -> float:
    """
    Supplementary metric: variance of the Hilbert envelope.

    This is kept for backwards-compatibility and potential inclusion in the
    supplementary material. It is not used as the headline metric.
    """
    x = _ensure_1d(x)
    analytic = hilbert(x)
    envelope = np.abs(analytic)
    return float(np.var(envelope))


def butterworth_lpf_gain_sq(delta_f: np.ndarray, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """
    Evaluate the squared magnitude response of an analog Butterworth low-pass filter.

    Parameters
    ----------
    delta_f:
        Frequency offset array (Hz) relative to the carrier.
    cutoff_hz:
        -3 dB cutoff frequency (Hz).
    order:
        Filter order (default 4, matching the time-domain I/Q demod).
    """
    cutoff = max(float(cutoff_hz), 1e-12)
    norm = np.maximum(1e-30, np.abs(delta_f) / cutoff)
    return 1.0 / (1.0 + norm ** (2 * order))


def one_sided_psd_from_omega(freq_hz: np.ndarray, psd_omega: np.ndarray) -> np.ndarray:
    """
    Convert a PSD defined over angular frequency ω to a one-sided PSD in Hz.

    Parameters
    ----------
    freq_hz:
        Frequency samples in Hz (non-negative).
    psd_omega:
        PSD evaluated at ω = 2πf (same shape as freq_hz), with units per rad/s.

    Returns
    -------
    np.ndarray
        One-sided PSD S₁(f) with units per Hz.
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    psd_omega = np.asarray(psd_omega, dtype=float)
    if freq_hz.shape != psd_omega.shape:
        raise ValueError("freq_hz and psd_omega must share the same shape.")
    psd_hz = np.array(psd_omega, dtype=float)
    positive = freq_hz > 0.0
    psd_hz[positive] *= 2.0
    return psd_hz


def narrowband_power_from_psd(
    freq_hz: np.ndarray,
    psd_one_sided: np.ndarray,
    center_hz: float,
    rel_bandwidth: float,
) -> float:
    """
    Integrate the one-sided PSD over a symmetric fractional bandwidth.

    Parameters
    ----------
    freq_hz:
        One-sided frequency grid (Hz).
    psd_one_sided:
        One-sided PSD samples corresponding to freq_hz.
    center_hz:
        Center frequency (Hz).
    rel_bandwidth:
        Fractional half-width (e.g., 0.3 => ±30%).
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    psd_one_sided = np.asarray(psd_one_sided, dtype=float)
    if freq_hz.size == 0:
        return 0.0
    beta = float(rel_bandwidth)
    if beta <= 0 or beta >= 1:
        raise ValueError("rel_bandwidth must lie in (0, 1).")
    lo = center_hz * (1.0 - beta)
    hi = center_hz * (1.0 + beta)
    mask = (freq_hz >= lo) & (freq_hz <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd_one_sided[mask], freq_hz[mask]))


def baseband_power_from_psd(
    freq_hz: np.ndarray,
    psd_one_sided: np.ndarray,
    center_hz: float,
    lp_frac: float,
    order: int = 4,
) -> float:
    """
    Compute the demodulated baseband variance using the analytic PSD.

    Parameters
    ----------
    freq_hz:
        One-sided frequency grid (Hz).
    psd_one_sided:
        One-sided PSD samples corresponding to freq_hz.
    center_hz:
        Carrier frequency f₃ (Hz).
    lp_frac:
        Low-pass cutoff expressed as a fraction of f₃.
    order:
        Butterworth filter order (default: 4).
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    psd_one_sided = np.asarray(psd_one_sided, dtype=float)
    if freq_hz.size == 0:
        return 0.0
    if not 0.0 < lp_frac < 1.0:
        raise ValueError("lp_frac must lie in (0, 1).")
    cutoff = lp_frac * center_hz
    gain_sq = butterworth_lpf_gain_sq(freq_hz - center_hz, cutoff, order=order)
    return float(np.trapezoid(gain_sq * psd_one_sided, freq_hz))


def band_average_from_psd(
    freq_hz: np.ndarray,
    psd_one_sided: np.ndarray,
    center_hz: float,
    rel_bandwidth: float,
) -> float:
    """
    Compute the band-averaged PSD around the specified carrier.
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    psd_one_sided = np.asarray(psd_one_sided, dtype=float)
    beta = float(rel_bandwidth)
    if beta <= 0 or beta >= 1:
        raise ValueError("rel_bandwidth must lie in (0, 1).")
    lo = center_hz * (1.0 - beta)
    hi = center_hz * (1.0 + beta)
    mask = (freq_hz >= lo) & (freq_hz <= hi)
    if not np.any(mask):
        return float("nan")
    integral = np.trapezoid(psd_one_sided[mask], freq_hz[mask])
    return float(integral / (2.0 * beta * center_hz))
