"""
Complete quantum hierarchy simulation combining all stages.
"""

import time as pytime

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from quantum_models import run_markovian_model, run_pseudomode_model


def main():
    print("=" * 60)
    print("QUANTUM HIERARCHICAL OSCILLATOR SIMULATION")
    print("Non-Markovian envelope amplification")
    print("=" * 60)

    print("\n[1/3] Running Markovian baseline...")
    start = pytime.time()
    times_m, x3_m, n3_m = run_markovian_model()
    print(f"  Completed in {pytime.time() - start:.1f}s")

    print("\n[2/3] Running structured bath (τ_B = 1e-3)...")
    start = pytime.time()
    times_s, x3_s, n3_s = run_pseudomode_model(tau_B=1e-3)
    print(f"  Completed in {pytime.time() - start:.1f}s")

    print("\n[3/3] Computing metrics...")

    freq_m, psd_m = signal.periodogram(x3_m, fs=1 / (times_m[1] - times_m[0]))
    freq_s, psd_s = signal.periodogram(x3_s, fs=1 / (times_s[1] - times_s[0]))

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
    print(f"  → Amplification: {e_s / e_m:.1f}×")

    print("\nLow-frequency band power (0.5-5 Hz):")
    print(f"  Markovian:  {band_m:.3e}")
    print(f"  Structured: {band_s:.3e}")
    print(f"  → Amplification: {band_s / band_m:.1f}×")

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times_m, x3_m, color="orange", alpha=0.8, linewidth=1.5)
    ax1.set_ylabel("x₃(t)", fontsize=12)
    ax1.set_title("Markovian (τ_B = 1e-5)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times_s, x3_s, color="dodgerblue", alpha=0.8, linewidth=1.5)
    ax2.set_ylabel("x₃(t)", fontsize=12)
    ax2.set_title("Structured (τ_B = 1e-3)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, :])
    psd_m_norm = psd_m / np.max(psd_m)
    psd_s_norm = psd_s / np.max(psd_s)
    ax3.loglog(freq_m, psd_m_norm, color="orange", linewidth=2.5, label="Markovian", alpha=0.8)
    ax3.loglog(freq_s, psd_s_norm, color="dodgerblue", linewidth=2.5, label="Structured", alpha=0.8)
    ax3.axvspan(0.5, 5, alpha=0.15, color="green", label="Envelope band (0.5-5)")
    ax3.set_xlabel("Frequency (arb. units)", fontsize=12)
    ax3.set_ylabel("Normalized PSD", fontsize=12)
    ax3.set_title("Power Spectral Density Comparison", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, which="both")
    ax3.set_xlim([0.1, 100])

    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(times_m, n3_m, color="orange", linewidth=2, label="Markovian", alpha=0.8)
    ax4.plot(times_s, n3_s, color="dodgerblue", linewidth=2, label="Structured", alpha=0.8)
    ax4.set_xlabel("Time (arb. units)", fontsize=12)
    ax4.set_ylabel("⟨n₃⟩ (energy)", fontsize=12)
    ax4.set_title("Slow Oscillator Energy Evolution", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.savefig("quantum_envelope_mechanism.png", dpi=200, bbox_inches="tight")
    print("\n✓ Figure saved as 'quantum_envelope_mechanism.png'")
    plt.show()

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

