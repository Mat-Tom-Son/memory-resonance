#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.metrics import baseband_power, narrowband_psd_power
from hierarchical_analysis import run_hierarchy, W1, W3


def compute_map(thetas: np.ndarray, alphas: np.ndarray, *, T: float, dt: float, burn_frac: float, tau_lp_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (R_ratio, Energy) for a grid of (Theta, alpha_feedback)."""
    n_theta, n_alpha = len(thetas), len(alphas)
    R = np.zeros((n_theta, n_alpha), dtype=float)
    E = np.zeros((n_theta, n_alpha), dtype=float)

    # Baseline per Theta at alpha=0
    baselines: list[float] = []
    for i, th in enumerate(thetas):
        tau_B = float(th / W1)
        times, x1, x2, x3 = run_hierarchy(
            tau_B=tau_B,
            T=T,
            dt=dt,
            calibration_mode="equal_carrier",
            noise_param=1.0,
            burn_in_fraction=burn_frac,
            alpha_feedback=0.0,
            tau_lp=0.0,
        )
        fs = 1.0 / (times[1] - times[0])
        bp = baseband_power(x3, fs_hz=fs, w3_rad_s=W3, lp_frac=0.2)
        baselines.append(max(bp, 1e-18))

    tau_lp = float(tau_lp_frac) * (1.0 / W1)
    for i, th in enumerate(thetas):
        tau_B = float(th / W1)
        for j, a in enumerate(alphas):
            times, x1, x2, x3 = run_hierarchy(
                tau_B=tau_B,
                T=T,
                dt=dt,
                calibration_mode="equal_carrier",
                noise_param=1.0,
                burn_in_fraction=burn_frac,
                alpha_feedback=float(a),
                tau_lp=tau_lp,
            )
            fs = 1.0 / (times[1] - times[0])
            bp = baseband_power(x3, fs_hz=fs, w3_rad_s=W3, lp_frac=0.2)
            R[i, j] = bp / baselines[i]
            E[i, j] = narrowband_psd_power(
                x1, fs_hz=fs, w3_rad_s=W1, rel_bandwidth=0.3, nperseg=4096, noverlap=2048
            )
    return R, E


def main() -> None:
    p = argparse.ArgumentParser(description="2D feedback map: Theta × alpha_feedback")
    p.add_argument("--theta-min", type=float, default=0.7)
    p.add_argument("--theta-max", type=float, default=1.4)
    p.add_argument("--n-theta", type=int, default=9)
    p.add_argument("--alpha-max", type=float, default=0.08)
    p.add_argument("--n-alpha", type=int, default=9)
    p.add_argument("--tau-lp-frac", type=float, default=0.3, help="LP time constant as fraction of 1/omega1")
    p.add_argument("--t-final", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--burn-frac", type=float, default=0.25)
    p.add_argument("--output", type=Path, default=Path("figures/figL_feedback_2d.pdf"))
    args = p.parse_args()

    thetas = np.linspace(args.theta_min, args.theta_max, args.n_theta, dtype=float)
    alphas = np.linspace(0.0, args.alpha_max, args.n_alpha, dtype=float)
    R, E = compute_map(
        thetas, alphas,
        T=float(args.t_final), dt=float(args.dt), burn_frac=float(args.burn_frac), tau_lp_frac=float(args.tau_lp_frac)
    )

    # Figure with two heatmaps and ridge overlay on R
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True, sharey=True)
    extent = [alphas.min(), alphas.max(), thetas.min(), thetas.max()]
    im0 = axes[0].imshow(R, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[0].set_title(r"$R_{env}$ ratio vs $\alpha_{fb}$")
    axes[0].set_xlabel(r"$\alpha_{fb}$")
    axes[0].set_ylabel(r"$\Theta$")
    fig.colorbar(im0, ax=axes[0], label=r"$R_{env}$ ratio")
    # Ridge (argmax along alpha per Theta)
    ridge_alpha = alphas[np.argmax(R, axis=1)]
    axes[0].plot(ridge_alpha, thetas, color="white", linewidth=1.2, linestyle="--", label="R ridge")
    axes[0].legend(loc="upper right")

    im1 = axes[1].imshow(E, origin="lower", aspect="auto", extent=extent, cmap="cividis")
    axes[1].set_title("Energy proxy (var x1)")
    axes[1].set_xlabel(r"$\alpha_{fb}$")
    fig.colorbar(im1, ax=axes[1], label="Energy (a.u.)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"✓ Saved {args.output}")


if __name__ == "__main__":
    main()
