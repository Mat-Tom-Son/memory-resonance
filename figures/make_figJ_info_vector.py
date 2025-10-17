#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.kernels import build_Kext_OU, estimate_Kint_from_H
from quant.gaussian_linear import build_model, psd as gl_psd
from quantum_models import W1, W2, W3, G12, G23, GAMMA1, GAMMA2, GAMMA3


def basis_exponentials(tau: np.ndarray, scales: list[float]) -> np.ndarray:
    Phi = []
    for s in scales:
        k = np.exp(-np.abs(tau) / s) / s
        # L1-normalize
        area = np.trapz(k, tau)
        k = k / area if area > 0 else k
        Phi.append(k)
    return np.stack(Phi, axis=1)  # shape (len(tau), K)


def project_vector(K: np.ndarray, tau: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    # Nonnegative least squares not required for small K; do normalized inner products
    # v_i = ∫ K(τ) φ_i(τ) dτ
    v = np.trapz(K[:, None] * Phi, tau, axis=0)
    # L2 normalize for cosine comparisons
    n = np.linalg.norm(v) or 1.0
    return v / n


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
    p = argparse.ArgumentParser(description="Information-vector cosine similarity vs performance")
    p.add_argument("--csv", type=Path, default=Path("results/adaptive/kerr08_det12_theta_grid7.csv"))
    p.add_argument("--output", type=Path, default=Path("figures/figJ_info_vector.pdf"))
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    theta = df["theta"].to_numpy()
    tau_B = df["tau_B"].to_numpy()
    Renv = df["R_env"].to_numpy()

    tau, Kint = estimate_Kint_linear_markovian()
    # Basis scales (seconds) around slow and intermediate timescales
    scales = [0.25 / W3, 0.5 / W3, 1.0 / W3, 2.0 / W3]
    Phi = basis_exponentials(tau, scales)
    vint = project_vector(Kint, tau, Phi)

    cosines = []
    for tB in tau_B:
        Kext = build_Kext_OU(tau, float(tB))
        vext = project_vector(Kext, tau, Phi)
        cosines.append(float(np.dot(vint, vext)))
    cosines = np.asarray(cosines, dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    # Panel A: cosine similarity vs R_env
    sc = ax1.scatter(cosines, Renv, c=theta, cmap="plasma", s=70, edgecolor="k", linewidth=0.5)
    ax1.set_xlabel("Cosine similarity (info vector)")
    ax1.set_ylabel(r"$R_{env}$")
    ax1.set_title("Similarity predicts response")
    ax1.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax1, label=r"$\Theta$")
    # Panel B: internal basis weights
    ax2.bar([f"τ={s*W3:.2f}/ω3" for s in scales], vint, color="tab:purple")
    ax2.set_ylabel("Internal weight (normalized)")
    ax2.set_title("Internal memory basis weights")
    for tick in ax2.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    print(f"✓ Saved {args.output}")


if __name__ == "__main__":
    main()

