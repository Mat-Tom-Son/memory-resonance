"""
Stage 3: Sweep bath memory parameter to map amplification regimes.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from quantum_models import run_pseudomode_model


def sweep_tau_B(tau_B_values):
    energies = []
    band_powers = []
    theta_values = []
    w1 = 1000.0

    for tau_B in tau_B_values:
        theta = w1 * tau_B
        theta_values.append(theta)

        print(f"\nRunning τ_B = {tau_B:.2e} (Θ = {theta:.3f})")
        times, x3, n3 = run_pseudomode_model(tau_B=tau_B)

        energies.append(np.mean(n3[-500:]))

        freq, psd = signal.periodogram(x3, fs=1 / (times[1] - times[0]))
        mask = (freq >= 0.5) & (freq <= 5)
        band_powers.append(np.sum(psd[mask]))

    return np.array(theta_values), np.array(energies), np.array(band_powers)


def main():
    tau_B_values = np.logspace(-5, -2, 10)
    theta, energies, band_powers = sweep_tau_B(tau_B_values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.semilogx(theta, energies / energies[0], "o-", linewidth=2, markersize=8)
    ax1.axvline(1.0, color="red", linestyle="--", alpha=0.5, label="Θ = 1")
    ax1.set_xlabel("Θ = ω₁ τ_B")
    ax1.set_ylabel("Energy amplification (normalized)")
    ax1.set_title("Slow-layer energy vs. bath memory")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.loglog(theta, band_powers / band_powers[0], "o-", linewidth=2, markersize=8)
    ax2.axvline(1.0, color="red", linestyle="--", alpha=0.5, label="Θ = 1")
    ax2.set_xlabel("Θ = ω₁ τ_B")
    ax2.set_ylabel("Band power amplification (normalized)")
    ax2.set_title("Envelope band power vs. bath memory")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("parameter_sweep.png", dpi=150)
    plt.show()

    print("\n" + "=" * 60)
    print("PARAMETER SWEEP SUMMARY")
    print("=" * 60)
    print(f"Peak amplification at Θ ≈ {theta[np.argmax(energies)]:.2f}")
    print(f"Max energy amplification: {np.max(energies) / energies[0]:.1f}×")
    print(f"Max band power amplification: {np.max(band_powers) / band_powers[0]:.1f}×")


if __name__ == "__main__":
    main()

