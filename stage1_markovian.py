"""
Stage 1: Basic three-oscillator model with Markovian damping.
"""

import matplotlib.pyplot as plt
import numpy as np

from quantum_models import run_markovian_model


def main():
    times, x3_traj, n3_traj = run_markovian_model()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(times, x3_traj, label="Markovian (white noise)", color="orange", linewidth=1.5)
    ax1.set_xlabel("Time (arb. units)")
    ax1.set_ylabel("x₃ (slow oscillator position)")
    ax1.set_title("Slow oscillator dynamics - Markovian bath")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, n3_traj, color="orange", linewidth=1.5)
    ax2.set_xlabel("Time (arb. units)")
    ax2.set_ylabel("⟨n₃⟩ (slow oscillator energy)")
    ax2.set_title("Slow oscillator energy - Markovian bath")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("markovian_baseline.png", dpi=150)
    plt.show()

    print(f"Mean slow-oscillator energy: {np.mean(n3_traj[-500:]):.3e}")


if __name__ == "__main__":
    main()

