#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.kernels import estimate_Kint_from_H
from quant.gaussian_linear import build_model, psd as gl_psd
from quantum_models import W1, W2, W3, G12, G23, GAMMA1, GAMMA2, GAMMA3


def ou_psd(omega: np.ndarray, tau_B: float, D: float = 1.0) -> np.ndarray:
    kappa = 1.0 / max(tau_B, 1e-12)
    return 2.0 * D * kappa / (omega**2 + kappa**2)


def gaussian_Rproxy(theta: float, omega: np.ndarray, win_center: float, rel_bw: float = 0.3, eps: float = 0.02, nrm_equal_carrier: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute δR/δSξ via a spectral bump experiment in a Gaussian linear proxy.

    Returns (omega_grid, grad), where grad(ωp) ≈ (R(ε) - R(0)) / (ε * ∫W).
    """
    tau_B = float(theta / W1)
    # Build linear Gaussian model and |H|^2 for x3 and fast-mode selector
    model = build_model(
        frequencies=[W1, W2, W3],
        couplings=[(0, 1, G12), (1, 2, G23)],
        dampings=[GAMMA1, GAMMA2, GAMMA3],
        nbar_list=[0.0, 0.0, 0.0],
        fast_mode=0,
        x_measure_mode=2,
    )
    omega_grid = np.linspace(0.0, 3.0 * W1, 2048, dtype=float)
    H2_x3 = np.array([gl_psd(model, model.x_selector, w) for w in omega_grid], dtype=float)
    H2_fast = np.array([gl_psd(model, model.fast_selector, w) for w in omega_grid], dtype=float)
    # Equal-carrier baseline: set C from Θ=1
    tau_B_ref = 1.0 / W1
    S_ref = ou_psd(omega_grid, tau_B_ref)
    idx_w1 = int(np.argmin(np.abs(omega_grid - W1)))
    C = H2_fast[idx_w1] * S_ref[idx_w1]

    # Window for envelope band around W3
    bw = rel_bw * win_center
    band = (omega_grid >= (win_center - bw)) & (omega_grid <= (win_center + bw))
    win_norm = float(np.trapezoid(np.ones_like(omega_grid[band], dtype=float), omega_grid[band]))

    # Baseline Sξ for this Θ with equal-carrier scaling
    S0 = ou_psd(omega_grid, tau_B)
    if nrm_equal_carrier:
        alpha = C / max(1e-18, H2_fast[idx_w1] * S0[idx_w1])
        S0 = alpha * S0

    # Baseline response proxy (band-limited variance for x3)
    R0 = float(np.trapezoid(H2_x3[band] * S0[band], omega_grid[band]))

    grad = []
    for i, wp in enumerate(omega_grid):
        # Narrow Gaussian bump at ωp with width σ = 0.02 ω1
        sigma = 0.02 * W1
        Wp = np.exp(-0.5 * ((omega_grid - wp) / max(1e-12, sigma)) ** 2)
        # Normalize bump area
        area_W = float(np.trapezoid(Wp, omega_grid))
        if area_W <= 0:
            grad.append(0.0)
            continue
        S_bumped = S0 * (1.0 + eps * Wp)
        if nrm_equal_carrier:
            # Renormalize to keep equal-carrier at ω1 fixed
            alpha_b = C / max(1e-18, H2_fast[idx_w1] * S_bumped[idx_w1])
            S_bumped = alpha_b * S_bumped
        Rb = float(np.trapezoid(H2_x3[band] * S_bumped[band], omega_grid[band]))
        grad.append((Rb - R0) / (eps * area_W))

    return omega_grid, np.asarray(grad, dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(description="True spectral sensitivity δR/δSξ via bump experiment (Gaussian proxy)")
    p.add_argument("--theta", type=float, default=0.90, help="Theta at which to compute gradient")
    p.add_argument("--output", type=Path, default=Path("figures/figM2_true_sensitivity.pdf"))
    args = p.parse_args()

    omega, grad = gaussian_Rproxy(args.theta, omega=np.array([]), win_center=W3, rel_bw=0.3, eps=0.02)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(omega, grad / max(1e-18, np.abs(grad).max()), color="tab:red", linewidth=1.4)
    ax.axvline(W1, color="#666666", linestyle="--", linewidth=0.9, label=r"$\omega_1$")
    ax.axvspan(W3 * (1 - 0.3), W3 * (1 + 0.3), color="#d9d9d9", alpha=0.35, label="slow band")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\delta R/\delta S_\xi(\omega)$ (norm)")
    ax.set_title(r"Spectral sensitivity at $\Theta=%.2f$" % args.theta)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"✓ Saved {args.output}")


if __name__ == "__main__":
    main()

