#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.metrics import baseband_power, narrowband_psd_power
from analysis.kernels import build_Kext_OU, estimate_Kint_from_H, overlap_O
from hierarchical_analysis import (
    W1,
    W3,
    run_hierarchy,
    ou_process_delayed,
)
from quant.gaussian_linear import build_model, psd as gl_psd
from quantum_models import W1 as QW1, W2, G12, G23, GAMMA1, GAMMA2, GAMMA3


def estimate_Kint_linear() -> tuple[np.ndarray, np.ndarray]:
    """Linear approximation of K_int via Gaussian model around operating point."""
    model = build_model(
        frequencies=[QW1, W2, W3],
        couplings=[(0, 1, G12), (1, 2, G23)],
        dampings=[GAMMA1, GAMMA2, GAMMA3],
        nbar_list=[0.0, 0.0, 0.0],
        fast_mode=0,
        x_measure_mode=2,
    )
    omega = np.linspace(0.0, 4.0 * W3, 1024, dtype=float)
    Sx = np.array([gl_psd(model, model.x_selector, w) for w in omega], dtype=float)
    tau, Kint = estimate_Kint_from_H(omega, Sx, window="tukey", alpha=0.2)
    return tau, Kint


def kext_with_delay(tau: np.ndarray, tau_B: float, delay: float) -> np.ndarray:
    """
    Delayed OU kernel approximation: shift OU kernel by positive delay.
    Re-normalize and clip negatives for numerical stability.
    """
    # Shift by delay: K(τ;τB,τd) ≈ K_OU(τ-τd; τB)
    k = build_Kext_OU(tau - float(delay), float(tau_B))
    k = np.maximum(k, 0.0)
    area = np.trapz(k, tau)
    if area > 0:
        k = k / area
    return k


def compute_map(
    *,
    thetas: np.ndarray,
    delays: np.ndarray,
    T: float,
    dt: float,
    burn_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (R_ratio map [len(thetas) x len(delays)], O map, Energy proxy map).
    R_ratio at each Θ is normalized by the no-delay baseline for that Θ.
    """
    tau, Kint = estimate_Kint_linear()
    n_theta = len(thetas)
    n_delay = len(delays)
    R = np.zeros((n_theta, n_delay), dtype=float)
    O = np.zeros((n_theta, n_delay), dtype=float)
    E = np.zeros((n_theta, n_delay), dtype=float)

    # Precompute baselines at zero delay per Θ
    baselines: list[float] = []
    for i, th in enumerate(thetas):
        tau_B = float(th / W1)
        xi0 = ou_process_delayed(T, dt, tau_B, delay=0.0, target_variance=1.0, seed=1234)
        times, x1, x2, x3 = run_hierarchy(
            tau_B=tau_B,
            T=T,
            dt=dt,
            calibration_mode="equal_carrier",
            noise_param=1.0,
            burn_in_fraction=burn_frac,
            noise_override=xi0,
        )
        fs = 1.0 / (times[1] - times[0])
        bp = baseband_power(x3, fs_hz=fs, w3_rad_s=W3, lp_frac=0.2)
        baselines.append(max(bp, 1e-18))

    for i, th in enumerate(thetas):
        tau_B = float(th / W1)
        for j, d in enumerate(delays):
            # Time-domain run with delayed OU noise
            xi = ou_process_delayed(T, dt, tau_B, delay=float(d), target_variance=1.0, seed=1234 + j)
            times, x1, x2, x3 = run_hierarchy(
                tau_B=tau_B,
                T=T,
                dt=dt,
                calibration_mode="equal_carrier",
                noise_param=1.0,
                burn_in_fraction=burn_frac,
                noise_override=xi,
            )
            fs = 1.0 / (times[1] - times[0])
            bp = baseband_power(x3, fs_hz=fs, w3_rad_s=W3, lp_frac=0.2)
            R[i, j] = bp / baselines[i]
            # Energy: band-limited fast-mode energy around ω1 (match narrowband config)
            E[i, j] = narrowband_psd_power(
                x1, fs_hz=fs, w3_rad_s=W1, rel_bandwidth=0.3, nperseg=4096, noverlap=2048
            )
            # Predictive overlap with delayed kernel
            Kext = kext_with_delay(tau, tau_B, delay=float(d))
            O[i, j] = float(np.trapz(Kext * Kint, tau))
        # Normalize row-wise overlaps for readability
        row = O[i, :]
        rng = max(1e-12, row.max() - row.min())
        O[i, :] = (row - row.min()) / rng
    return R, O, E


def main() -> None:
    p = argparse.ArgumentParser(description="2D memory (τB × delay) map with overlap, response, energy")
    p.add_argument("--theta-min", type=float, default=0.7)
    p.add_argument("--theta-max", type=float, default=1.4)
    p.add_argument("--n-theta", type=int, default=9)
    p.add_argument("--delays", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--t-final", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--burn-frac", type=float, default=0.25)
    p.add_argument("--output", type=Path, default=Path("figures/figK_memory_2d.pdf"))
    args = p.parse_args()

    thetas = np.linspace(args.theta_min, args.theta_max, args.n_theta, dtype=float)
    # delays are fractions of 1/W1
    delays = np.asarray(args.delays, dtype=float) * (1.0 / W1)
    R, O, E = compute_map(
        thetas=thetas,
        delays=delays,
        T=float(args.t_final),
        dt=float(args.dt),
        burn_frac=float(args.burn_frac),
    )

    # Prepare figure: three heatmaps side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True, sharey=True)
    extent = [delays.min() * W1, delays.max() * W1, thetas.min(), thetas.max()]  # delay axis in units of 1/W1

    im0 = axes[0].imshow(R, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[0].set_title(r"$R_{env}$ ratio vs no-delay")
    axes[0].set_xlabel(r"Delay $\tau_d \cdot \omega_1$ (fraction of $1/\omega_1$)")
    axes[0].set_ylabel(r"$\Theta = \omega_1 \tau_B$")
    fig.colorbar(im0, ax=axes[0], label=r"$R_{env}$ ratio")

    im1 = axes[1].imshow(O, origin="lower", aspect="auto", extent=extent, cmap="magma")
    axes[1].set_title("Kernel overlap O (norm)")
    axes[1].set_xlabel(r"Delay $\tau_d \cdot \omega_1$")
    fig.colorbar(im1, ax=axes[1], label="O (norm)")

    im2 = axes[2].imshow(E, origin="lower", aspect="auto", extent=extent, cmap="cividis")
    axes[2].set_title("Energy proxy (var x1)")
    axes[2].set_xlabel(r"Delay $\tau_d \cdot \omega_1$")
    fig.colorbar(im2, ax=axes[2], label="Energy (a.u.)")

    # Ridge overlays: argmax along delay for each Θ on R and O panels
    delay_grid = delays * W1
    ridg_R = delay_grid[np.argmax(R, axis=1)]
    ridg_O = delay_grid[np.argmax(O, axis=1)]
    axes[0].plot(ridg_R, thetas, color="white", linewidth=1.2, linestyle="--", label="R ridge")
    axes[1].plot(ridg_O, thetas, color="white", linewidth=1.2, linestyle="--", label="O ridge")
    for ax in axes[:2]:
        ax.legend(loc="upper right")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"✓ Saved {args.output}")


if __name__ == "__main__":
    main()
