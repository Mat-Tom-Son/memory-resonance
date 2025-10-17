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


def estimate_O_Idot(theta: float) -> tuple[float, float]:
    tau_B = float(theta / W1)
    model = build_model(
        frequencies=[W1, W2, W3],
        couplings=[(0, 1, G12), (1, 2, G23)],
        dampings=[GAMMA1, GAMMA2, GAMMA3],
        nbar_list=[0.0, 0.0, 0.0],
        fast_mode=0,
        x_measure_mode=2,
    )
    omega = np.linspace(0.0, 4.0 * W3, 1024, dtype=float)
    H2 = np.array([gl_psd(model, model.x_selector, w) for w in omega], dtype=float)
    tau, Kint = estimate_Kint_from_H(omega, H2, window="tukey", alpha=0.2)
    O = overlap_O(tau, Kint, [tau_B], kext_builder=build_Kext_OU)[0]
    # Info rate
    kappa = 1.0 / max(tau_B, 1e-12)
    Sxi = 2.0 * kappa / (omega**2 + kappa**2)
    Idot = float(np.trapezoid(np.log1p(H2 * Sxi), omega) / (2.0 * np.pi))
    return O, Idot


def main() -> None:
    p = argparse.ArgumentParser(description="Frontier table for S, C, M(+), M(null)")
    p.add_argument("--csv-classical", type=Path, default=Path("results/theta_sweep_today.csv"))
    p.add_argument("--csv-classC", type=Path, default=Path("results/classical_parametric_mod03.csv"))
    p.add_argument("--csv-mplus", type=Path, nargs="+", default=[Path("results/adaptive/kerr08_det12_theta_grid7.csv")])
    p.add_argument("--csv-mnull", type=Path, default=Path("results/quantum_eqheat_sweep_for_figB.csv"))
    p.add_argument("--output", type=Path, default=Path("figures/figTable_frontier.pdf"))
    args = p.parse_args()

    rows = []
    # Class S (baseline): take Θ*=1.0 by construction
    th_S = 1.0
    O_S, I_S = estimate_O_Idot(th_S)
    E_S = np.nan  # omitted for S due to lack of time-domain fast energy; proxy not needed
    rows.append(["Class S", f"{th_S:.2f}", f"{O_S:.3f}", "—", f"{I_S:.3e}", "—", "—"])  # η’s omitted

    # Class C: pick peak from classical_parametric_mod03.csv
    dfC = pd.read_csv(args.csv_classC)
    idxC = int(np.nanargmax(dfC["band_ratio"].to_numpy()))
    th_C = float(dfC.loc[idxC, "theta"])
    R_C = float(dfC.loc[idxC, "band_ratio"])  # R_env proxy
    O_C, I_C = estimate_O_Idot(th_C)
    rows.append(["Class C", f"{th_C:.2f}", f"{O_C:.3f}", "—", f"{I_C:.3e}", "—", f"{R_C:.3f}"])

    # Class M(+): peak from adaptive repeats (mean across CSVs if multiple)
    df_list = [pd.read_csv(p) for p in args.csv_mplus]
    df0 = df_list[0]
    theta = df0["theta"].to_numpy()
    R_stack = np.stack([d["R_env"].to_numpy() for d in df_list], axis=0)
    R_mean = R_stack.mean(axis=0)
    idxM = int(np.nanargmax(R_mean))
    th_M = float(theta[idxM])
    R_M = float(R_mean[idxM])
    E_M = float(np.stack([d.get("n1_mean", pd.Series(np.nan, index=range(len(theta)))).to_numpy() for d in df_list], axis=0).mean(axis=0)[idxM])
    O_M, I_M = estimate_O_Idot(th_M)
    etaI_M = I_M / max(E_M, 1e-18)
    etaR_M = max(0.0, R_M - 1.0) / max(E_M, 1e-18)
    rows.append(["Class M (+)", f"{th_M:.2f}", f"{O_M:.3f}", f"{E_M:.3e}", f"{I_M:.3e}", f"{etaI_M:.3e}", f"{etaR_M:.3e}"])

    # Class M (null): choose Θ=1.0 from null CSV
    dfN = pd.read_csv(args.csv_mnull)
    if "theta" in dfN.columns and "R_env" in dfN.columns:
        # pick closest to Θ=1
        idxN = int(np.nanargmin(np.abs(dfN["theta"].to_numpy() - 1.0)))
        th_N = float(dfN.loc[idxN, "theta"])
        R_N = float(dfN.loc[idxN, "R_env"]) if "R_env" in dfN.columns else 1.0
    else:
        th_N, R_N = 1.0, 1.0
    O_N, I_N = estimate_O_Idot(th_N)
    rows.append(["Class M (null)", f"{th_N:.2f}", f"{O_N:.3f}", "—", f"{I_N:.3e}", "—", f"{R_N:.3f}"])

    # Render table
    fig, ax = plt.subplots(figsize=(7.0, 1.8))
    ax.axis("off")
    cols = ["Pillar", "Peak Θ", "O", "E", r"$\dot I$", r"$\eta_I$", r"$\eta_R$"]
    table = ax.table(cellText=rows, colLabels=cols, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    print(f"✓ Saved {args.output}")


if __name__ == "__main__":
    main()

