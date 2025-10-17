#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.kernels import build_Kext_OU, estimate_Kint_from_H, overlap_O
from quant.gaussian_linear import build_model, psd as gl_psd
from quantum_models import W1, W2, W3, G12, G23, GAMMA1, GAMMA2, GAMMA3


def estimate_Kint_linear_markovian() -> tuple[np.ndarray, np.ndarray]:
    model = build_model(
        frequencies=[W1, W2, W3],
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


def main() -> None:
    p = argparse.ArgumentParser(description="Memory (overlap) vs Energy map colored by performance")
    p.add_argument("--csv", type=Path, default=Path("results/adaptive/kerr08_det12_theta_grid7.csv"))
    p.add_argument("--output", type=Path, default=Path("figures/figI_memory_energy.pdf"))
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    theta = df["theta"].to_numpy()
    tau_B = df["tau_B"].to_numpy()
    Renv = df["R_env"].to_numpy()
    # Energy proxy: fast-mode occupancy (n1_mean) if present; else env_power
    if "n1_mean" in df.columns:
        Energy = df["n1_mean"].to_numpy()
    elif "env_power" in df.columns:
        Energy = df["env_power"].to_numpy()
    else:
        Energy = np.ones_like(Renv)

    # Compute kernel overlap O(τB)
    tau, Kint = estimate_Kint_linear_markovian()
    Ovals = overlap_O(tau, Kint, tau_B, kext_builder=build_Kext_OU)
    Ovals_n = (Ovals - Ovals.min()) / max(1e-12, (Ovals.max() - Ovals.min()))

    # Approximate Gaussian information rate for linear+Gaussian proxy
    # SNR(ω) = |H(ω)|^2 S_xi(ω) / N0, here N0=1 (normalized units)
    # Under equal-carrier, the ω1 lobe is fixed; any τB dependence in Idot reflects memory structure.
    omega = np.linspace(0.0, 4.0 * W3, 1024, dtype=float)
    # Reuse the same model used for K_int estimation
    model = build_model(
        frequencies=[W1, W2, W3],
        couplings=[(0, 1, G12), (1, 2, G23)],
        dampings=[GAMMA1, GAMMA2, GAMMA3],
        nbar_list=[0.0, 0.0, 0.0],
        fast_mode=0,
        x_measure_mode=2,
    )
    H2 = np.array([gl_psd(model, model.x_selector, w) for w in omega], dtype=float)
    def ou_psd(omega: np.ndarray, tau_B_val: float, D: float = 1.0) -> np.ndarray:
        kappa = 1.0 / max(tau_B_val, 1e-12)
        return 2.0 * D * kappa / (omega**2 + kappa**2)
    I_dot = []
    for tB in tau_B:
        Sxi = ou_psd(omega, float(tB))
        snr = H2 * Sxi  # N0=1
        I_dot.append(float(np.trapezoid(np.log1p(snr), omega) / (2.0 * np.pi)))
    I_dot = np.asarray(I_dot, dtype=float)

    # Compute efficiencies
    # Energy E: prefer n1_mean (quantum); else use provided Energy
    E = Energy.copy()
    # Guard against tiny energy
    E_safe = np.maximum(E, 1e-18)
    eta_I = I_dot / E_safe
    eta_R = np.maximum(0.0, Renv - 1.0) / E_safe

    # Plot Memory vs Energy scatter colored by R_env
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sc = ax.scatter(Ovals_n, Energy, c=Renv, cmap="viridis", s=70, edgecolor="k", linewidth=0.5)
    for i, th in enumerate(theta):
        ax.annotate(f"{th:.2f}", (Ovals_n[i], Energy[i]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    cbar = fig.colorbar(sc, ax=ax, label=r"$R_{env}$")
    ax.set_xlabel("Kernel overlap O (normalized)")
    ax.set_ylabel("Energy (fast-mode occupancy)")
    ax.set_title("Memory × Energy map (color = performance)")
    ax.grid(True, alpha=0.3)

    # Inset: efficiencies η_I and η_R vs Θ
    inset = ax.inset_axes([0.62, 0.12, 0.35, 0.38])
    inset2 = inset.twinx()
    inset.plot(theta, eta_R, color="tab:purple", linewidth=1.2, label=r"$\eta_R$")
    inset2.plot(theta, eta_I, color="tab:green", linewidth=1.0, linestyle=":", label=r"$\eta_I$")
    inset.set_xlabel(r"$\Theta$", fontsize=8)
    inset.set_ylabel(r"$\eta_R$", fontsize=8)
    inset2.set_ylabel(r"$\eta_I$", fontsize=8)
    inset.tick_params(axis="both", labelsize=7)
    inset.grid(True, alpha=0.2)
    # Small legend
    lines, labels = inset.get_legend_handles_labels()
    lines2, labels2 = inset2.get_legend_handles_labels()
    inset.legend(lines + lines2, labels + labels2, fontsize=7, loc="upper left")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"✓ Saved {args.output}")


if __name__ == "__main__":
    main()
