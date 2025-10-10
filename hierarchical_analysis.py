"""
Hierarchical Oscillator Envelope Amplification
Production version with exact OU discretization and dual calibration modes.

Author: Mat Thompson
Date: October 2025
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.stats import bootstrap
from tqdm import tqdm

warnings.filterwarnings("ignore")

W1 = 1000.0
W2 = 10.0
W3 = 1.0
G12 = 0.05
G23 = 0.05
GAMMA1 = 0.1
GAMMA2 = 0.05
GAMMA3 = 0.01
DEFAULT_DT = 1e-4
DEFAULT_SWEEP_DURATION = 60.0
DEFAULT_NOISE_LEVEL = 1.0
DEFAULT_N_SEEDS = 6
ENVELOPE_BAND = (0.5, 5.0)


# ============================================================================
# EXACT OU PROCESS (dt-independent)
# ============================================================================

def ou_process_exact(
    T: float,
    dt: float,
    tau_B: float,
    target_variance: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Exact discrete-time OU process via AR(1) representation.

    Ensures steady-state variance is target_variance regardless of dt.
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T / dt)
    xi = np.zeros(n_steps)

    rho = np.exp(-dt / tau_B)
    var_eta = target_variance * (1 - rho**2)

    eta = np.sqrt(var_eta) * np.random.randn(n_steps - 1)

    for i in range(n_steps - 1):
        xi[i + 1] = rho * xi[i] + eta[i]

    return xi


def calibrate_variance_for_carrier_power(
    tau_B: float,
    S_target: float,
    omega_carrier: float,
) -> float:
    """
    Compute target variance to achieve specified spectral density at ω₁.
    """
    return S_target * (1 + (omega_carrier * tau_B) ** 2) / (2 * tau_B)


# ============================================================================
# HIERARCHICAL MODEL
# ============================================================================

def run_hierarchy(
    tau_B: float,
    T: float = 200.0,
    dt: float = 1e-4,
    calibration_mode: str = "equal_variance",
    noise_param: float = 1.0,
    burn_in_fraction: float = 0.25,
    seed: int | None = None,
    noise_override: np.ndarray | None = None,
    return_noise: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Three weakly coupled oscillators driven by a calibrated OU bath.
    """
    if seed is not None:
        np.random.seed(seed)

    w1, w2, w3 = W1, W2, W3
    g12, g23 = G12, G23
    gamma1, gamma2, gamma3 = GAMMA1, GAMMA2, GAMMA3

    if calibration_mode == "equal_variance":
        target_var = noise_param
    elif calibration_mode == "equal_carrier":
        target_var = calibrate_variance_for_carrier_power(tau_B, noise_param, w1)
    else:
        raise ValueError(f"Unknown calibration_mode: {calibration_mode}")

    n_steps = int(T / dt)
    if noise_override is not None:
        xi = np.asarray(noise_override, dtype=float).ravel()
        if xi.size < n_steps:
            raise ValueError("Provided noise_override is shorter than required steps.")
        xi = xi[:n_steps]
    else:
        xi = ou_process_exact(T, dt, tau_B, target_variance=target_var, seed=seed)

    x1, v1 = np.zeros(n_steps), np.zeros(n_steps)
    x2, v2 = np.zeros(n_steps), np.zeros(n_steps)
    x3, v3 = np.zeros(n_steps), np.zeros(n_steps)

    for i in range(n_steps - 1):
        a1 = -2 * gamma1 * v1[i] - w1**2 * x1[i] + g12 * (x2[i] - x1[i]) + xi[i]
        v1[i + 1] = v1[i] + a1 * dt
        x1[i + 1] = x1[i] + v1[i + 1] * dt

        a2 = (
            -2 * gamma2 * v2[i]
            - w2**2 * x2[i]
            + g12 * (x1[i] - x2[i])
            + g23 * (x3[i] - x2[i])
        )
        v2[i + 1] = v2[i] + a2 * dt
        x2[i + 1] = x2[i] + v2[i + 1] * dt

        a3 = -2 * gamma3 * v3[i] - w3**2 * x3[i] + g23 * (x2[i] - x3[i])
        v3[i + 1] = v3[i] + a3 * dt
        x3[i + 1] = x3[i] + v3[i + 1] * dt

    burn_idx = int(burn_in_fraction * n_steps)
    times = np.arange(n_steps) * dt

    results = (times[burn_idx:], x1[burn_idx:], x2[burn_idx:], x3[burn_idx:])
    if return_noise:
        return (*results, xi[burn_idx:].copy())
    return results


# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

def compute_psd_welch(
    x: np.ndarray,
    dt: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Power spectral density via Welch's method with Hann window.
    """
    fs = 1.0 / dt
    if nperseg is None:
        nperseg = min(len(x) // 8, 4096)
    if noverlap is None:
        noverlap = nperseg // 2

    freq, psd = signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        detrend="constant",
    )
    return freq, psd


def compute_band_power(freq: np.ndarray, psd: np.ndarray, f_low: float, f_high: float) -> float:
    """Integrate PSD over frequency band."""
    mask = (freq >= f_low) & (freq <= f_high)
    if not np.any(mask):
        return 0.0
    return np.trapz(psd[mask], freq[mask])


def bootstrap_ci(
    data: np.ndarray,
    statistic=np.median,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval."""
    result = bootstrap(
        (data,),
        statistic,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="BCa",
        random_state=42,
    )
    return result.confidence_interval.low, result.confidence_interval.high


# ============================================================================
# PRODUCTION SWEEP (CLI)
# ============================================================================

def _logspace(theta_min: float, theta_max: float, n_points: int) -> np.ndarray:
    if theta_min <= 0 or theta_max <= 0:
        raise ValueError("theta_min and theta_max must be positive for log sweep.")
    if theta_max <= theta_min:
        raise ValueError("theta_max must be greater than theta_min.")
    return np.logspace(np.log10(theta_min), np.log10(theta_max), n_points)


def run_classical_sweep(
    theta_min: float,
    theta_max: float,
    n_points: int,
    calibrations: Sequence[str],
    output_dir: str,
    n_seeds: int = DEFAULT_N_SEEDS,
    duration: float = DEFAULT_SWEEP_DURATION,
    dt: float = DEFAULT_DT,
    noise_param: float = DEFAULT_NOISE_LEVEL,
    envelope_band: tuple[float, float] = ENVELOPE_BAND,
    burn_in_fraction: float = 0.25,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Run a Θ sweep for the requested calibration modes and write summary artifacts.

    Returns
    -------
    tuple
        (calibration_stats, theta_values)
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    if n_seeds < 3:
        raise ValueError("n_seeds must be at least 3 for bootstrap stability.")
    if not calibrations:
        raise ValueError("At least one calibration must be provided.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    theta_values = _logspace(theta_min, theta_max, n_points)
    tau_values = theta_values / W1
    seeds = [int(10_000 + s) for s in range(n_seeds)]

    results: dict[str, dict[str, np.ndarray]] = {}
    summary_rows: list[tuple[float, str, float, float, float]] = []

    print("\n" + "=" * 70)
    print("CLASSICAL ENVELOPE SWEEP")
    print("=" * 70)
    print(f"Θ range: [{theta_values[0]:.3f}, {theta_values[-1]:.3f}] with {n_points} points")
    print(f"Calibrations: {', '.join(calibrations)}")
    print(f"Seeds per Θ: {n_seeds} (bootstrap)")
    print(f"Simulation duration: {duration} (dt = {dt:.1e})")
    print(f"Envelope band: {envelope_band[0]} - {envelope_band[1]}")
    print(f"Output directory: {output_path}")

    for calibration in calibrations:
        print("\n" + "-" * 70)
        print(f"Calibration: {calibration}")
        band_matrix = np.zeros((n_points, n_seeds), dtype=np.float64)

        with tqdm(total=n_points * n_seeds, desc=f"{calibration:>16}", ncols=90) as pbar:
            for theta_idx, tau_B in enumerate(tau_values):
                for seed_idx, seed in enumerate(seeds):
                    _, _, _, x3 = run_hierarchy(
                        tau_B=tau_B,
                        T=duration,
                        dt=dt,
                        calibration_mode=calibration,
                        noise_param=noise_param,
                        burn_in_fraction=burn_in_fraction,
                        seed=seed,
                    )
                    freq, psd = compute_psd_welch(x3, dt)
                    band_power = compute_band_power(freq, psd, envelope_band[0], envelope_band[1])
                    band_matrix[theta_idx, seed_idx] = band_power
                    pbar.update(1)

        baseline = band_matrix[0, :]
        baseline = np.where(baseline > 0, baseline, np.nan)
        r_env_matrix = band_matrix / baseline

        r_env_mean = np.nanmean(r_env_matrix, axis=1)
        r_env_ci_low = np.zeros(n_points)
        r_env_ci_high = np.zeros(n_points)
        band_mean = np.nanmean(band_matrix, axis=1)
        band_std = np.nanstd(band_matrix, axis=1)

        for idx, theta in enumerate(theta_values):
            samples = r_env_matrix[idx, :]
            samples = samples[~np.isnan(samples)]
            if samples.size == 0:
                ci_low, ci_high = np.nan, np.nan
            else:
                ci_low, ci_high = bootstrap_ci(np.asarray(samples), statistic=np.mean, n_resamples=5_000)

            r_env_ci_low[idx] = ci_low
            r_env_ci_high[idx] = ci_high
            summary_rows.append((theta, calibration, r_env_mean[idx], ci_low, ci_high))

        peak_idx = int(np.nanargmax(r_env_mean))
        peak_theta = theta_values[peak_idx]
        peak_amp = r_env_mean[peak_idx]
        print(f"  Peak R_env ≈ {peak_amp:.2f} at Θ ≈ {peak_theta:.2f}")

        np.save(output_path / f"band_power_{calibration}.npy", band_matrix)

        results[calibration] = {
            "r_env_mean": r_env_mean,
            "r_env_ci_low": r_env_ci_low,
            "r_env_ci_high": r_env_ci_high,
            "band_mean": band_mean,
            "band_std": band_std,
        }

    summary_path = output_path / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Theta", "calibration", "R_env_mean", "R_env_ci_low", "R_env_ci_high"])
        for theta, calibration, mean_val, ci_low, ci_high in summary_rows:
            writer.writerow([f"{theta:.12g}", calibration, f"{mean_val:.12g}", f"{ci_low:.12g}", f"{ci_high:.12g}"])

    np.save(output_path / "theta_values.npy", theta_values)

    print(f"\n✓ Summary written to {summary_path}")
    return results, theta_values


def _select_example_thetas(theta_values: np.ndarray) -> list[float]:
    """Pick illustrative Θ values (min, near Θ=1, max)."""
    selections = [theta_values[0]]
    target = 1.0
    idx = int(np.argmin(np.abs(np.log(theta_values) - np.log(target))))
    if theta_values[idx] not in selections:
        selections.append(theta_values[idx])
    if theta_values[-1] not in selections:
        selections.append(theta_values[-1])
    return selections


def save_classical_resonance_figure(
    theta_values: np.ndarray,
    calibration_results: dict[str, dict[str, np.ndarray]],
    calibrations: Sequence[str],
    output_dir: Path,
    dt: float = DEFAULT_DT,
    duration: float = DEFAULT_SWEEP_DURATION / 2,
    noise_param: float = DEFAULT_NOISE_LEVEL,
    envelope_band: tuple[float, float] = ENVELOPE_BAND,
    burn_in_fraction: float = 0.25,
) -> None:
    """Create summary figure with R_env curves, envelope power, and PSD examples."""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9], hspace=0.32, wspace=0.3)

    ax_main = fig.add_subplot(gs[0, :])
    ax_band = fig.add_subplot(gs[1, 0])
    ax_psd = fig.add_subplot(gs[1, 1])

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(calibrations), 3)))

    for idx, calibration in enumerate(calibrations):
        data = calibration_results[calibration]
        color = colors[idx % len(colors)]
        ax_main.semilogx(theta_values, data["r_env_mean"], label=calibration, color=color, linewidth=2.3)
        ax_main.fill_between(
            theta_values,
            data["r_env_ci_low"],
            data["r_env_ci_high"],
            alpha=0.2,
            color=color,
        )

        band_mean = data["band_mean"]
        band_std = data["band_std"]
        ax_band.loglog(theta_values, band_mean, color=color, linewidth=2.0, label=calibration)
        lower = np.clip(band_mean - band_std, a_min=band_mean.min() * 0.2, a_max=None)
        ax_band.fill_between(theta_values, lower, band_mean + band_std, alpha=0.15, color=color)

    ax_main.set_xlabel("Θ = ω₁ τ_B", fontsize=12)
    ax_main.set_ylabel("R_env (normalized envelope power)", fontsize=12)
    ax_main.set_title("Envelope Amplification vs Bath Memory", fontsize=14, fontweight="bold")
    ax_main.grid(True, alpha=0.25)
    ax_main.legend(fontsize=11)

    ax_band.set_xlabel("Θ = ω₁ τ_B", fontsize=12)
    ax_band.set_ylabel("Envelope power (0.5-5)", fontsize=12)
    ax_band.set_title("Envelope Band Power vs Θ", fontsize=13, fontweight="bold")
    ax_band.grid(True, alpha=0.25, which="both")
    ax_band.legend(fontsize=10)

    example_thetas = _select_example_thetas(theta_values)
    linestyles = ["-", "--", ":"]
    for calibration_idx, calibration in enumerate(calibrations):
        base_color = colors[calibration_idx % len(colors)]
        for example_idx, theta in enumerate(example_thetas):
            tau_B = theta / W1
            _, _, _, x3 = run_hierarchy(
                tau_B=tau_B,
                T=duration,
                dt=dt,
                calibration_mode=calibration,
                noise_param=noise_param,
                burn_in_fraction=burn_in_fraction,
                seed=42,
            )
            freq, psd = compute_psd_welch(x3, dt)
            psd_norm = psd / np.max(psd)
            label = f"{calibration}, Θ={theta:.2f}"
            ax_psd.loglog(
                freq,
                psd_norm,
                linewidth=1.8,
                alpha=0.8,
                label=label,
                color=base_color,
                linestyle=linestyles[example_idx % len(linestyles)],
            )

    ax_psd.axvspan(envelope_band[0], envelope_band[1], color="mediumseagreen", alpha=0.15, label="Envelope band")
    ax_psd.set_xlabel("Frequency", fontsize=12)
    ax_psd.set_ylabel("Normalized PSD", fontsize=12)
    ax_psd.set_title("Representative Power Spectra", fontsize=13, fontweight="bold")
    ax_psd.set_xlim([0.1, 200])
    ax_psd.grid(True, alpha=0.25, which="both")
    ax_psd.legend(fontsize=9)

    fig.suptitle("Classical Hierarchical Envelope Resonance", fontsize=16, fontweight="bold")
    figure_path = output_dir / "classical_envelope_resonance.png"
    plt.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Figure saved: {figure_path}")


# ============================================================================
# EXPERIMENT 1: CONTROLLED COMPARISON
# ============================================================================

def experiment_1_comparison(calibration_mode: str = "equal_variance") -> None:
    """Compare Markovian vs structured bath with matched noise power."""
    print("=" * 70)
    print(f"EXPERIMENT 1: Controlled Comparison ({calibration_mode})")
    print("=" * 70)

    T = 200.0
    dt = 1e-4
    noise_param = 1.0

    print("\nParameters:")
    print(f"  Simulation time: T = {T:.1f} (≈{T/(2*np.pi):.1f} slow periods)")
    print(f"  Time step: dt = {dt:.0e} (stability: dt·ω₁ = {dt * W1:.2f})")
    print(f"  Calibration: {calibration_mode}")
    print(f"  Noise parameter: {noise_param}")

    print("\n[1/2] Markovian (τ_B = 1e-5, Θ = 0.01)...")
    t_m, _, _, x3_m = run_hierarchy(
        tau_B=1e-5,
        T=T,
        dt=dt,
        calibration_mode=calibration_mode,
        noise_param=noise_param,
        seed=42,
    )

    print("[2/2] Structured (τ_B = 1e-3, Θ = 1.0)...")
    t_s, _, _, x3_s = run_hierarchy(
        tau_B=1e-3,
        T=T,
        dt=dt,
        calibration_mode=calibration_mode,
        noise_param=noise_param,
        seed=42,
    )

    print("\nComputing PSDs (Welch: Hann window, 50% overlap)...")
    freq_m, psd_m = compute_psd_welch(x3_m, dt)
    freq_s, psd_s = compute_psd_welch(x3_s, dt)

    var_m = np.var(x3_m)
    var_s = np.var(x3_s)

    band_m = compute_band_power(freq_m, psd_m, 0.5, 5.0)
    band_s = compute_band_power(freq_s, psd_s, 0.5, 5.0)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nSlow-layer variance:")
    print(f"  Markovian:  {var_m:.3e}")
    print(f"  Structured: {var_s:.3e}")
    print(f"  → Amplification: {var_s / var_m:.1f}×")

    print(f"\nEnvelope band power (0.5-5 Hz):")
    print(f"  Markovian:  {band_m:.3e}")
    print(f"  Structured: {band_s:.3e}")
    print(f"  → Amplification: {band_s / band_m:.1f}×")

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    show_long = 50.0
    n_long = int(show_long / dt)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_m[:n_long], x3_m[:n_long], color="orange", alpha=0.7, linewidth=0.8)
    ax1.set_ylabel("x₃(t)", fontsize=11)
    ax1.set_title("Markovian (τ_B=1e-5, Θ=0.01) - Long View", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_s[:n_long], x3_s[:n_long], color="dodgerblue", alpha=0.7, linewidth=0.8)
    ax2.set_ylabel("x₃(t)", fontsize=11)
    ax2.set_title("Structured (τ_B=1e-3, Θ=1.0) - Long View", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    show_zoom = 5.0
    n_zoom = int(show_zoom / dt)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t_m[:n_zoom], x3_m[:n_zoom], color="orange", alpha=0.8, linewidth=1.2)
    ax3.set_ylabel("x₃(t)", fontsize=11)
    ax3.set_title("Markovian - Detail", fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t_s[:n_zoom], x3_s[:n_zoom], color="dodgerblue", alpha=0.8, linewidth=1.2)
    ax4.set_ylabel("x₃(t)", fontsize=11)
    ax4.set_title("Structured - Detail (note envelope modulation)", fontsize=10)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, :])
    psd_m_norm = psd_m / np.max(psd_m)
    psd_s_norm = psd_s / np.max(psd_s)
    ax5.loglog(freq_m, psd_m_norm, color="orange", linewidth=2.5, label="Markovian", alpha=0.8)
    ax5.loglog(freq_s, psd_s_norm, color="dodgerblue", linewidth=2.5, label="Structured", alpha=0.8)
    ax5.axvspan(0.5, 5, alpha=0.15, color="green", label="Envelope band (0.5-5)")
    ax5.set_xlabel("Frequency (arb. units)", fontsize=12)
    ax5.set_ylabel("Normalized PSD", fontsize=12)
    ax5.set_title(
        "Power Spectral Density (Welch: Hann window, 50% overlap, detrended)",
        fontsize=12,
        fontweight="bold",
    )
    ax5.legend(fontsize=11, loc="upper right")
    ax5.grid(True, alpha=0.3, which="both")
    ax5.set_xlim([0.1, 200])

    ax6 = fig.add_subplot(gs[3, 0])
    window_size = len(t_m) // 10
    windows_m = [
        np.var(x3_m[i : i + window_size]) for i in range(0, len(x3_m) - window_size, window_size // 2)
    ]
    windows_s = [
        np.var(x3_s[i : i + window_size]) for i in range(0, len(x3_s) - window_size, window_size // 2)
    ]
    ax6.plot(windows_m, "o-", color="orange", label="Markovian", alpha=0.7, markersize=5)
    ax6.plot(windows_s, "o-", color="dodgerblue", label="Structured", alpha=0.7, markersize=5)
    ax6.set_xlabel("Window index", fontsize=11)
    ax6.set_ylabel("Var(x₃)", fontsize=11)
    ax6.set_title("Stationarity Check", fontsize=11, fontweight="bold")
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[3, 1])
    metrics = ["Variance", "Band Power\n(0.5-5)"]
    ratios = [var_s / var_m, band_s / band_m]
    colors = ["steelblue", "mediumseagreen"]

    bars = ax7.bar(metrics, ratios, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax7.axhline(1, color="red", linestyle="--", linewidth=2, alpha=0.5, label="No effect")
    ax7.set_ylabel("Structured / Markovian", fontsize=12)
    ax7.set_title("Amplification Ratios", fontsize=11, fontweight="bold")
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3, axis="y")

    for bar, ratio in zip(bars, ratios):
        ax7.text(
            bar.get_x() + bar.get_width() / 2.0,
            ratio,
            f"{ratio:.1f}×",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.savefig(f"experiment1_{calibration_mode}.png", dpi=200, bbox_inches="tight")
    print(f"\n✓ Figure saved: experiment1_{calibration_mode}.png")
    plt.show()


# ============================================================================
# EXPERIMENT 2: τ_B SWEEP WITH ERROR BARS
# ============================================================================

def experiment_2_sweep(
    calibration_mode: str = "equal_variance",
    n_seeds: int = 16,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Sweep τ_B to map envelope amplification versus Θ = ω₁ τ_B.
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: Bath Memory Sweep ({calibration_mode})")
    print("=" * 70)

    tau_B_values = np.logspace(-5, -1, 25)
    theta_values = W1 * tau_B_values

    T = 100.0
    dt = 1e-4
    noise_param = 1.0

    print(f"\nSettings:")
    print(f"  τ_B range: [{tau_B_values[0]:.1e}, {tau_B_values[-1]:.1e}]")
    print(f"  Θ range: [{theta_values[0]:.2f}, {theta_values[-1]:.2f}]")
    print(f"  Seeds per point: {n_seeds}")
    print(f"  Total runs: {len(tau_B_values) * n_seeds}")

    variances = np.zeros((len(tau_B_values), n_seeds))
    band_powers = np.zeros((len(tau_B_values), n_seeds))

    print("\nRunning sweep...")
    for i, tau_B in enumerate(tqdm(tau_B_values, desc="Progress")):
        for j in range(n_seeds):
            _, _, _, x3 = run_hierarchy(
                tau_B,
                T=T,
                dt=dt,
                calibration_mode=calibration_mode,
                noise_param=noise_param,
                seed=100 * i + j,
            )

            variances[i, j] = np.var(x3)
            freq, psd = compute_psd_welch(x3, dt)
            band_powers[i, j] = compute_band_power(freq, psd, 0.5, 5.0)

    var_median = np.median(variances, axis=1)
    var_low = np.percentile(variances, 2.5, axis=1)
    var_high = np.percentile(variances, 97.5, axis=1)

    band_median = np.median(band_powers, axis=1)
    band_low = np.percentile(band_powers, 2.5, axis=1)
    band_high = np.percentile(band_powers, 97.5, axis=1)

    var_norm = var_median / var_median[0]
    var_norm_low = var_low / var_median[0]
    var_norm_high = var_high / var_median[0]

    band_norm = band_median / band_median[0]
    band_norm_low = band_low / band_median[0]
    band_norm_high = band_high / band_median[0]

    peak_idx = np.argmax(band_norm)
    theta_peak = theta_values[peak_idx]
    amp_peak = band_norm[peak_idx]

    peak_thetas: list[float] = []
    for _ in range(1000):
        resample_idx = np.random.choice(n_seeds, n_seeds, replace=True)
        resample_medians = np.median(band_powers[:, resample_idx], axis=1)
        resample_norm = resample_medians / resample_medians[0]
        peak_thetas.append(theta_values[np.argmax(resample_norm)])

    theta_peak_ci = np.percentile(peak_thetas, [2.5, 97.5])

    print("\n" + "=" * 70)
    print("SWEEP RESULTS")
    print("=" * 70)
    print(f"\nPeak amplification: {amp_peak:.1f}× ")
    print(f"  at Θ = {theta_peak:.2f} (95% CI: [{theta_peak_ci[0]:.2f}, {theta_peak_ci[1]:.2f}])")
    print(f"\nPredicted resonance: Θ ≈ 1.0")
    print(f"Deviation: {abs(theta_peak - 1.0) / 1.0:.1%}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.semilogx(
        theta_values,
        var_norm,
        "o-",
        color="steelblue",
        linewidth=2.5,
        markersize=6,
        label=f"Median (n={n_seeds})",
    )
    ax1.fill_between(theta_values, var_norm_low, var_norm_high, alpha=0.25, color="steelblue", label="95% CI")
    ax1.axvline(1.0, color="red", linestyle="--", linewidth=2.5, alpha=0.7, label="Θ = 1 (predicted)")
    ax1.axvline(theta_peak, color="green", linestyle=":", linewidth=2.5, alpha=0.8, label=f"Peak (Θ={theta_peak:.2f})")
    ax1.set_xlabel("Θ = ω₁ τ_B", fontsize=13)
    ax1.set_ylabel("Variance amplification", fontsize=13)
    ax1.set_title(f"Slow-Layer Variance vs Bath Memory\n({calibration_mode})", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([theta_values[0] * 0.8, theta_values[-1] * 1.2])

    ax2.loglog(
        theta_values,
        band_norm,
        "o-",
        color="mediumseagreen",
        linewidth=2.5,
        markersize=6,
        label=f"Median (n={n_seeds})",
    )
    ax2.fill_between(theta_values, band_norm_low, band_norm_high, alpha=0.25, color="mediumseagreen", label="95% CI")
    ax2.axvline(1.0, color="red", linestyle="--", linewidth=2.5, alpha=0.7, label="Θ = 1 (predicted)")
    ax2.axvline(theta_peak, color="green", linestyle=":", linewidth=2.5, alpha=0.8, label=f"Peak (Θ={theta_peak:.2f})")
    ax2.set_xlabel("Θ = ω₁ τ_B", fontsize=13)
    ax2.set_ylabel("Band power amplification (0.5-5)", fontsize=13)
    ax2.set_title(f"Envelope Band Power vs Bath Memory\n({calibration_mode})", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_xlim([theta_values[0] * 0.8, theta_values[-1] * 1.2])

    plt.tight_layout()
    plt.savefig(f"experiment2_sweep_{calibration_mode}.png", dpi=200, bbox_inches="tight")
    print(f"\n✓ Figure saved: experiment2_sweep_{calibration_mode}.png")
    plt.show()

    return theta_values, band_norm, theta_peak


# ============================================================================
# RUN ALL / CLI ENTRY
# ============================================================================

def run_demo(calibration_mode: str = "equal_variance") -> None:
    print("\n" + "=" * 70)
    print("HIERARCHICAL OSCILLATOR ANALYSIS - PRODUCTION VERSION")
    print("Exact OU discretization + dual calibration modes")
    print("=" * 70)

    print(f"\nCalibration mode: {calibration_mode}")
    print("  equal_variance: match ⟨ξ²⟩ across τ_B")
    print("  equal_carrier:  match S_ξ(ω₁) across τ_B")

    experiment_1_comparison(calibration_mode=calibration_mode)
    _, _, peak = experiment_2_sweep(calibration_mode=calibration_mode, n_seeds=16)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nKey result: Peak envelope amplification at Θ = {peak:.2f}")
    print(f"           (predicted Θ = 1.0, error = {abs(peak - 1.0):.1%})")
    print("\nReady for quantum replication.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hierarchical oscillator analysis (classical baseline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "sweep"],
        default="demo",
        help="demo: replicate paper figures; sweep: produce CSVs/plots for pipeline",
    )
    parser.add_argument(
        "--theta_min",
        type=float,
        default=0.01,
        help="Minimum Θ value (only for sweep mode).",
    )
    parser.add_argument(
        "--theta_max",
        type=float,
        default=100.0,
        help="Maximum Θ value (only for sweep mode).",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=40,
        help="Number of Θ samples between min and max (sweep mode).",
    )
    parser.add_argument(
        "--calibrations",
        nargs="+",
        default=["equal_variance"],
        help="Calibration modes to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/classical_sweep",
        help="Directory for sweep outputs (sweep mode).",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Number of stochastic seeds per Θ for bootstrap (sweep mode).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_SWEEP_DURATION,
        help="Simulation duration per run in arbitrary time units (sweep mode).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help="Integrator time step (sweep mode).",
    )
    parser.add_argument(
        "--noise_param",
        type=float,
        default=DEFAULT_NOISE_LEVEL,
        help="Noise calibration parameter (sweep mode).",
    )
    parser.add_argument(
        "--burn_in",
        type=float,
        default=0.25,
        help="Fraction of samples discarded as burn-in (sweep mode).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    calibrations = args.calibrations or ["equal_variance"]

    if args.mode == "sweep":
        sweep_results, theta_values = run_classical_sweep(
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            n_points=args.n_points,
            calibrations=calibrations,
            output_dir=args.output_dir,
            n_seeds=args.n_seeds,
            duration=args.duration,
            dt=args.dt,
            noise_param=args.noise_param,
            burn_in_fraction=args.burn_in,
        )
        save_classical_resonance_figure(
            theta_values=theta_values,
            calibration_results=sweep_results,
            calibrations=calibrations,
            output_dir=Path(args.output_dir),
            dt=args.dt,
            duration=min(args.duration, max(args.duration / 2.0, 20.0)),
            noise_param=args.noise_param,
            burn_in_fraction=args.burn_in,
        )
    else:
        run_demo(calibration_mode=calibrations[0])


if __name__ == "__main__":
    main()
