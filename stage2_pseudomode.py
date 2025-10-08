"""
Stage 2: Add pseudomode for structured (non-Markovian) noise.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from quantum_models import run_markovian_model, run_pseudomode_model


def main():
    times_m, x3_m, n3_m = run_markovian_model()
    times_s, x3_s, n3_s = run_pseudomode_model(tau_B=1e-3)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(times_m, x3_m, label="Markovian (τ_B=1e-5)", color="orange", alpha=0.8)
    axes[0, 0].set_ylabel("x₃(t)")
    axes[0, 0].set_title("Slow oscillator position - Markovian")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(times_s, x3_s, label="Structured (τ_B=1e-3)", color="dodgerblue", alpha=0.8)
    axes[0, 1].set_ylabel("x₃(t)")
    axes[0, 1].set_title("Slow oscillator position - Structured bath")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    freq_m, psd_m = signal.periodogram(x3_m, fs=1 / (times_m[1] - times_m[0]))
    freq_s, psd_s = signal.periodogram(x3_s, fs=1 / (times_s[1] - times_s[0]))

    psd_m_norm = psd_m / np.max(psd_m)
    psd_s_norm = psd_s / np.max(psd_s)

    axes[1, 0].loglog(freq_m, psd_m_norm, color="orange", linewidth=2, label="Markovian")
    axes[1, 0].loglog(freq_s, psd_s_norm, color="dodgerblue", linewidth=2, label="Structured")
    axes[1, 0].set_xlabel("Frequency")
    axes[1, 0].set_ylabel("Normalized PSD")
    axes[1, 0].set_title("Power spectra comparison")
    axes[1, 0].axvspan(0.5, 5, alpha=0.2, color="green", label="Envelope band")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0.1, 100])

    axes[1, 1].plot(times_m, n3_m, color="orange", label="Markovian", linewidth=2)
    axes[1, 1].plot(times_s, n3_s, color="dodgerblue", label="Structured", linewidth=2)
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("⟨n₃⟩")
    axes[1, 1].set_title("Slow oscillator energy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("quantum_comparison.png", dpi=150)
    plt.show()

    e_m = np.mean(n3_m[-500:])
    e_s = np.mean(n3_s[-500:])

    mask_m = (freq_m >= 0.5) & (freq_m <= 5)
    mask_s = (freq_s >= 0.5) & (freq_s <= 5)
    band_m = np.sum(psd_m[mask_m])
    band_s = np.sum(psd_s[mask_s])

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("\nMean slow-oscillator energy:")
    print(f"  Markovian:  {e_m:.3e}")
    print(f"  Structured: {e_s:.3e}")
    print(f"  Ratio:      {e_s / e_m:.1f}×")

    print("\nLow-frequency band power (0.5-5):")
    print(f"  Markovian:  {band_m:.3e}")
    print(f"  Structured: {band_s:.3e}")
    print(f"  Ratio:      {band_s / band_m:.1f}×")


if __name__ == "__main__":
    main()

