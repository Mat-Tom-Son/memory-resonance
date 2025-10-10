"""
Stage 4: Mixed pseudomode bath to probe phase diversity.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from quantum_models import run_pseudomode_model


def compute_phases(signal: np.ndarray, burn_in: float) -> np.ndarray:
    n_burn = int(len(signal) * burn_in)
    trimmed = signal[n_burn:] - np.mean(signal[n_burn:])
    analytic = hilbert(trimmed)
    phases = np.angle(analytic)
    return phases


def aggregate_hist(phases: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, float]:
    counts, _ = np.histogram(phases, bins=bins, density=False)
    counts = counts.astype(float)
    probability = counts / counts.sum() if counts.sum() > 0 else counts
    entropy = 0.0
    non_zero = probability > 0
    if np.any(non_zero):
        entropy = float(-np.sum(probability[non_zero] * np.log(probability[non_zero])))
    density = probability / (bins[1] - bins[0])
    return density, entropy


def build_mixed_signal(signals: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    mix = np.zeros_like(signals[0])
    for w, sig in zip(weights, signals):
        mix += w * sig
    return mix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mixed pseudomode phase diversity experiment (Stage 4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tauB_list",
        nargs="+",
        type=float,
        required=True,
        help="List of bath correlation times to combine (expect two values).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Number of random mixtures to generate.",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="equal_carrier",
        help="Calibration mode passed to the pseudomode model.",
    )
    parser.add_argument(
        "--burn_in",
        type=float,
        default=0.3,
        help="Fraction of samples treated as burn-in before phase analysis.",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=36,
        help="Number of bins for the phase histogram.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for mixture weights.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/mixed_modes",
        help="Directory to write figures and CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.tauB_list) < 2:
        raise ValueError("Provide at least two τ_B values for the mixed pseudomode experiment.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    stored_signals: list[np.ndarray] = []
    stored_phases: dict[float, np.ndarray] = {}
    for tau_B in args.tauB_list:
        times, x3, _ = run_pseudomode_model(tau_B=tau_B, calibration_mode=args.calibration)
        stored_signals.append(x3)
        stored_phases[tau_B] = compute_phases(x3, burn_in=args.burn_in)

    bins = np.linspace(-np.pi, np.pi, args.n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    entropies: list[tuple[str, str, float]] = []
    single_histograms: dict[float, np.ndarray] = {}
    for tau_B, phases in stored_phases.items():
        density, entropy = aggregate_hist(phases, bins)
        single_histograms[tau_B] = density
        entropies.append((f"single_tau={tau_B:.3f}", f"{tau_B:.6g}", entropy))

    mixed_phases = []
    for _ in range(args.n_samples):
        weights = rng.dirichlet(alpha=np.ones(len(stored_signals)))
        mix_signal = build_mixed_signal(stored_signals, weights)
        phases = compute_phases(mix_signal, burn_in=args.burn_in)
        mixed_phases.append(phases)
    if mixed_phases:
        mixed_phases_arr = np.concatenate(mixed_phases)
    else:
        mixed_phases_arr = np.array([])

    mixed_density, mixed_entropy = aggregate_hist(mixed_phases_arr, bins)
    entropies.append(
        (
            "mixed_tau=" + "+".join(f"{tau:.3f}" for tau in args.tauB_list),
            "",
            mixed_entropy,
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(args.tauB_list)))
    for color, tau_B in zip(colors, args.tauB_list):
        axes[0].plot(bin_centers, single_histograms[tau_B], linewidth=2.2, label=f"τ_B={tau_B}", color=color)
    axes[0].set_title("Single-τ envelope phase density", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Phase (rad)")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    axes[1].plot(bin_centers, mixed_density, color="firebrick", linewidth=2.5)
    axes[1].set_title("Mixed-τ envelope phase density", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Phase (rad)")
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlim([-np.pi, np.pi])
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    fig.suptitle("Phase diversity: single vs mixed pseudomode baths", fontsize=15, fontweight="bold")
    figure_path = output_dir / "phase_histogram.png"
    fig.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Phase histogram saved to {figure_path}")

    entropy_path = output_dir / "phase_entropy.csv"
    with entropy_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", "tau_B", "entropy"])
        for config, tau_label, entropy in entropies:
            writer.writerow([config, tau_label, f"{entropy:.12g}"])
    print(f"✓ Phase entropy table saved to {entropy_path}")


if __name__ == "__main__":
    main()
