from __future__ import annotations

import numpy as np

from analysis.metrics import baseband_power, narrowband_psd_power


def _am_signal(depth: float, duration: float = 2.0, fs: float = 2000.0) -> tuple[np.ndarray, float]:
    t = np.arange(int(duration * fs)) / fs
    carrier_hz = 5.0
    carrier_w = 2.0 * np.pi * carrier_hz
    mod_hz = 0.5
    envelope = 1.0 + depth * np.sin(2.0 * np.pi * mod_hz * t)
    signal = envelope * np.cos(carrier_w * t)
    return signal.astype(float), carrier_w


def test_baseband_power_monotonic_in_am_depth() -> None:
    fs = 2000.0
    lp_frac = 0.2
    sig_lo, w3 = _am_signal(0.05, fs=fs)
    sig_hi, _ = _am_signal(0.25, fs=fs)

    power_lo = baseband_power(sig_lo, fs_hz=fs, w3_rad_s=w3, lp_frac=lp_frac)
    power_hi = baseband_power(sig_hi, fs_hz=fs, w3_rad_s=w3, lp_frac=lp_frac)

    assert power_hi > power_lo * 1.5  # expect clear separation


def test_narrowband_psd_power_tracks_carrier() -> None:
    fs = 2000.0
    sig, w3 = _am_signal(0.1, fs=fs)
    band_power = narrowband_psd_power(
        sig,
        fs_hz=fs,
        w3_rad_s=w3,
        rel_bandwidth=0.3,
        nperseg=1024,
        noverlap=512,
    )
    assert band_power > 0.0
