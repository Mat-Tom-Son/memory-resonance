#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot classical parametric modulation sweep (Class C)")
    parser.add_argument("--csv", type=Path, default=Path("results/classical_parametric_mod03.csv"))
    parser.add_argument("--csv-baseline", type=Path, default=Path("results/classical_parametric_mod00.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures/figD_parametric_classC.pdf"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["band_ratio", "band_ratio_surrogate"])
    theta = df["theta"].values
    ratio = df["band_ratio"].values
    ratio_surr = df["band_ratio_surrogate"].values
    sem = df["band_sem"].values / df["band_mean"].values * ratio
    sem_surr = df["band_sem_surrogate"].values / df["band_mean_surrogate"].values * ratio_surr

    baseline_ratio = baseline_theta = baseline_sem = None
    if args.csv_baseline.exists():
        df_base = pd.read_csv(args.csv_baseline)
        df_base = df_base.replace([np.inf, -np.inf], np.nan).dropna(subset=["band_ratio"])
        baseline_theta = df_base["theta"].values
        baseline_ratio = df_base["band_ratio"].values
        baseline_sem = df_base["band_sem"].values / df_base["band_mean"].values * baseline_ratio

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.fill_between([0.7, 1.4], 0, 2, color="lightgray", alpha=0.3, zorder=0, label="MR band")
    ax.errorbar(theta, ratio, yerr=sem, marker="o", color="tab:blue", label="OU + modulation", linewidth=1.5)
    ax.errorbar(theta, ratio_surr, yerr=sem_surr, marker="s", color="tab:orange", label="PSD-matched surrogate", linewidth=1.2)
    if baseline_ratio is not None:
        ax.errorbar(baseline_theta, baseline_ratio, yerr=baseline_sem, marker="^", linestyle="--", color="tab:gray", label="OU (no modulation)", linewidth=1.0)

    ax.set_xlabel(r"$\Theta = \omega_1 \tau_B$")
    ax.set_ylabel(r"$R_\mathrm{env}$ (norm. envelope power)")
    ax.set_title("Class C diagnostic: PSD surrogate fails under parametric modulation")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(theta.min() - 0.05, theta.max() + 0.05)
    ymax = max(ratio.max(), ratio_surr.max())
    if baseline_ratio is not None:
        ymax = max(ymax, baseline_ratio.max())
    ax.set_ylim(0.9, ymax * 1.1)
    ax.legend(loc="upper left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"✓ Saved figure to {args.output}")


if __name__ == "__main__":
    main()
