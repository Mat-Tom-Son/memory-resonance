"""
Complete quantum hierarchy simulation combining all stages and BLP diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import time as pytime
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from quantum_models import (
    W1,
    compute_BLP_non_markovianity,
    run_markovian_model,
    run_pseudomode_model,
)

ENVELOPE_BAND = (0.5, 5.0)


def compute_band_power(times: np.ndarray, series: np.ndarray) -> float:
    dt = float(times[1] - times[0])
    freq, psd = signal.periodogram(series, fs=1.0 / dt)
    mask = (freq >= ENVELOPE_BAND[0]) & (freq <= ENVELOPE_BAND[1])
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freq[mask]))


def tail_average(values: np.ndarray, fraction: float = 0.3) -> float:
    start = max(int(len(values) * (1.0 - fraction)), 0)
    return float(np.mean(values[start:]))


def run_demo(calibration_mode: str = "equal_carrier", tau_B: float = 1e-3) -> None:
    print("=" * 60)
    print("QUANTUM HIERARCHICAL OSCILLATOR SIMULATION")
    print("Non-Markovian envelope amplification")
    print("=" * 60)
    print(f"Calibration: {calibration_mode}")

    print("\n[1/3] Running Markovian baseline...")
    start = pytime.time()
    times_m, x3_m, n3_m = run_markovian_model(calibration_mode=calibration_mode)
    print(f"  Completed in {pytime.time() - start:.1f}s")

    print(f"\n[2/3] Running structured bath (τ_B = {tau_B:.1e})...")
    start = pytime.time()
    times_s, x3_s, n3_s = run_pseudomode_model(tau_B=tau_B, calibration_mode=calibration_mode)
    print(f"  Completed in {pytime.time() - start:.1f}s")

    print("\n[3/3] Computing metrics...")

    band_m = compute_band_power(times_m, x3_m)
    band_s = compute_band_power(times_s, x3_s)
    e_m = tail_average(n3_m)
    e_s = tail_average(n3_s)

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
    ax1.set_title("Markovian bath", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times_s, x3_s, color="dodgerblue", alpha=0.8, linewidth=1.5)
    ax2.set_ylabel("x₃(t)", fontsize=12)
    ax2.set_title(f"Pseudomode bath (τ_B={tau_B:.1e})", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, :])
    freq_m, psd_m = signal.periodogram(x3_m, fs=1.0 / (times_m[1] - times_m[0]))
    freq_s, psd_s = signal.periodogram(x3_s, fs=1.0 / (times_s[1] - times_s[0]))
    ax3.loglog(freq_m, psd_m / np.max(psd_m), color="orange", linewidth=2.5, label="Markovian", alpha=0.8)
    ax3.loglog(freq_s, psd_s / np.max(psd_s), color="dodgerblue", linewidth=2.5, label="Structured", alpha=0.8)
    ax3.axvspan(ENVELOPE_BAND[0], ENVELOPE_BAND[1], alpha=0.15, color="green", label="Envelope band")
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


def load_theta_sweep_summary(
    summary_path: Path,
    calibration: str,
    cutoff: int,
    stride: int = 1,
) -> list[dict[str, float]]:
    data: list[dict[str, float]] = []
    with summary_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model") != "pseudomode":
                continue
            if row.get("calibration") != calibration:
                continue
            try:
                row_cutoff = int(float(row["cutoff"]))
            except (KeyError, ValueError):
                continue
            if row_cutoff != cutoff:
                continue
            try:
                entry = {
                    "Theta": float(row["Theta"]),
                    "tau_B": float(row["tau_B"]),
                    "R_env": float(row["R_env"]),
                    "N": int(float(row["N"])),
                    "N_pseudo": int(float(row["N_pseudo"])),
                }
            except (ValueError, KeyError):
                continue
            data.append(entry)

    data.sort(key=lambda item: item["Theta"])
    if stride > 1:
        data = data[::stride]
    return data


def measure_blp_correlation(
    summary_path: Path,
    output_path: Path,
    calibration: str,
    cutoff: int,
    times: np.ndarray,
    state_delta: float,
    stride: int,
) -> float:
    entries = load_theta_sweep_summary(summary_path, calibration, cutoff, stride=stride)
    if not entries:
        raise RuntimeError(f"No pseudomode entries found for calibration='{calibration}' cutoff={cutoff} in {summary_path}")

    results: list[tuple[float, float, float, float]] = []
    for entry in entries:
        params = {
            "tau_B": entry["tau_B"],
            "N": entry["N"],
            "N_pseudo": entry["N_pseudo"],
            "times": times,
            "calibration_mode": calibration,
            "state_delta": state_delta,
        }
        n_blp, _, _ = compute_BLP_non_markovianity(params=params, states=None)
        results.append((entry["Theta"], entry["tau_B"], entry["R_env"], n_blp))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Theta", "tau_B", "R_env", "N_BLP"])
        for row in results:
            writer.writerow([f"{row[0]:.12g}", f"{row[1]:.12g}", f"{row[2]:.12g}", f"{row[3]:.12g}"])
    print(f"✓ BLP correlation table saved to {output_path}")

    theta_vals = np.array([row[0] for row in results])
    r_env_vals = np.array([row[2] for row in results])
    n_blp_vals = np.array([row[3] for row in results])

    corr = float(np.corrcoef(r_env_vals, n_blp_vals)[0, 1]) if len(results) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        r_env_vals,
        n_blp_vals,
        c=theta_vals,
        cmap="plasma",
        s=70,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("R_env (band power ratio)", fontsize=12)
    ax.set_ylabel("N_BLP", fontsize=12)
    ax.set_title(f"Envelope Gain vs BLP (calibration={calibration}, cutoff={cutoff})\nρ = {corr:.3f}", fontsize=13)
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Θ = ω₁ τ_B", fontsize=11)

    figure_path = output_path.with_name("R_env_vs_BLP.png")
    fig.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ BLP scatter saved to {figure_path}")
    return corr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantum hierarchy driver and diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--measure",
        choices=["BLP"],
        help="Optional diagnostic to compute instead of running the demo.",
    )
    parser.add_argument(
        "--theta_sweep",
        type=str,
        help="Summary CSV from stage3_parameter_sweep.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/blp_correlation.csv",
        help="Output CSV for diagnostic measurements.",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="equal_carrier",
        help="Calibration mode to analyse.",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=8,
        help="Pseudomode cutoff to analyse for BLP (must exist in sweep summary).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sub-sample factor when reading the sweep (stride=2 uses every 2nd Θ).",
    )
    parser.add_argument(
        "--state_delta",
        type=float,
        default=0.08,
        help="Separation between initial states for BLP computation.",
    )
    parser.add_argument(
        "--t_final",
        type=float,
        default=2.5,
        help="Total evolution time for diagnostic runs.",
    )
    parser.add_argument(
        "--n_time",
        type=int,
        default=600,
        help="Number of solver time points for diagnostics.",
    )
    parser.add_argument(
        "--tau_B",
        type=float,
        default=1e-3,
        help="Structured bath correlation time for the demo mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.measure == "BLP":
        if not args.theta_sweep:
            raise ValueError("--theta_sweep must be provided when --measure BLP is set.")
        summary_path = Path(args.theta_sweep)
        output_path = Path(args.output)
        times = np.linspace(0, args.t_final, args.n_time)
        corr = measure_blp_correlation(
            summary_path=summary_path,
            output_path=output_path,
            calibration=args.calibration,
            cutoff=args.cutoff,
            times=times,
            state_delta=args.state_delta,
            stride=args.stride,
        )
        print(f"Computed Pearson correlation between R_env and N_BLP: {corr:.3f}")
    else:
        run_demo(calibration_mode=args.calibration, tau_B=args.tau_B)


if __name__ == "__main__":
    main()
