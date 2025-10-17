#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.kernels import build_Kext_OU, estimate_Kint_from_H, overlap_O, build_Kext_flat
from quantum_models import W1, W2, W3, G12, G23, GAMMA1, GAMMA2, GAMMA3
from quant.gaussian_linear import build_model, psd as gl_psd


def estimate_Kint_linear_markovian(w_slow: float = W3) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate K_int by linear Gaussian model around operating point (Markovian bath).
    Uses the drift/diffusion structure to compute PSD of x3, then windowed IFT.
    """
    # Build a basic linear Markovian model (3 modes)
    model = build_model(
        frequencies=[W1, W2, w_slow],
        couplings=[(0, 1, G12), (1, 2, G23)],
        dampings=[GAMMA1, GAMMA2, GAMMA3],
        nbar_list=[0.0, 0.0, 0.0],
        fast_mode=0,
        x_measure_mode=2,
    )
    # Frequency grid for PSD (around the slow band to limit ringing)
    wmin = 0.0
    wmax = 4.0 * w_slow
    omega = np.linspace(wmin, wmax, 1024, dtype=float)
    Sx = np.array([gl_psd(model, model.x_selector, w) for w in omega], dtype=float)
    tau, Kint = estimate_Kint_from_H(omega, Sx, window="tukey", alpha=0.2)
    return tau, Kint


def main() -> None:
    p = argparse.ArgumentParser(description="Kernel overlap O(τ_B) vs R_env(τ_B) overlay")
    p.add_argument("--quantum-csv", type=Path, default=Path("results/adaptive/kerr08_det12_theta_grid7.csv"))
    p.add_argument("--output", type=Path, default=Path("figures/figH_kernel_overlap.pdf"))
    args = p.parse_args()

    df = pd.read_csv(args.quantum_csv)
    theta = df["theta"].to_numpy()
    tau_B = df["tau_B"].to_numpy()
    Renv = df["R_env"].to_numpy()

    # Estimate K_int once from linear Markovian approximation
    tau, Kint = estimate_Kint_linear_markovian(W3)
    Ovals = overlap_O(tau, Kint, tau_B, kext_builder=build_Kext_OU)
    # Null overlap with flat kernel (band-equalized null)
    O0 = overlap_O(tau, Kint, tau_B, kext_builder=lambda tt, _tB: build_Kext_flat(tt))
    # Rescale O to be comparable on the plot (unitless overlay)
    Ovals_n = (Ovals - Ovals.min()) / max(1e-12, (Ovals.max() - Ovals.min()))
    O0_n = (O0 - O0.min()) / max(1e-12, (O0.max() - O0.min()))
    Renv_n = (Renv - Renv.min()) / max(1e-12, (Renv.max() - Renv.min()))

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(theta, Renv, marker="^", color="tab:purple", label=r"$R_{env}(\Theta)$")
    ax.plot(theta, Renv_n, linestyle=":", color="tab:purple", alpha=0.4, label="R_env (norm)")
    ax.plot(theta, Ovals_n, marker="o", color="tab:green", label="Kernel overlap O (norm)")
    ax.plot(theta, O0_n, linestyle="--", color="#888888", label="Null O0 (flat kernel)")
    ax.set_xlabel(r"$\Theta = \omega_1 \tau_B$")
    ax.set_ylabel("Response / Overlap (a.u.)")
    ax.set_title("Kernel overlap vs envelope response")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"✓ Saved {args.output}")


if __name__ == "__main__":
    main()
