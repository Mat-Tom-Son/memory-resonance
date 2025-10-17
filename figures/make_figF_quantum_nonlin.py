#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum equal-carrier sweep with detuning + Kerr")
    parser.add_argument("--csv", type=Path, nargs="+", default=[Path("results/quantum_nonlin/kerr02_tol001.csv")])
    parser.add_argument("--csv-surrogate", type=Path, default=Path("results/quantum_nonlin/kerr00_tol001.csv"))
    parser.add_argument("--csv-negative", type=Path, default=Path("results/quantum_nonlin/kerr02_tol001_neg.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures/figF_quantum_nonlin.pdf"))
    args = parser.parse_args()

    dfs = [pd.read_csv(p).replace([np.inf, -np.inf], np.nan) for p in args.csv]
    df0 = dfs[0]
    theta = df0["theta"].to_numpy()
    ratios = np.stack([d["R_env"].to_numpy() for d in dfs], axis=0)
    ratio = ratios.mean(axis=0)
    ratio_sem = ratios.std(axis=0, ddof=1) / np.sqrt(max(1, ratios.shape[0])) if ratios.shape[0] > 1 else np.zeros_like(ratio)
    # Auxiliary diagnostics (mean across repeats if present)
    def _mean_col(col: str) -> np.ndarray:
        cols = []
        for d in dfs:
            if col in d.columns:
                cols.append(d[col].to_numpy())
        if not cols:
            return np.full_like(ratio, np.nan)
        return np.stack(cols, axis=0).mean(axis=0)
    env_ratio = _mean_col("R_env_periodogram")
    hilbert_ratio = _mean_col("R_env_hilbert")

    surrogate_ratio = None
    theta_s = None
    if args.csv_surrogate and args.csv_surrogate.exists():
        df_s = pd.read_csv(args.csv_surrogate)
        theta_s = df_s["theta"].values
        surrogate_ratio = df_s["R_env"].values

    neg_ratio = None
    theta_neg = None
    if args.csv_negative and args.csv_negative.exists():
        df_neg = pd.read_csv(args.csv_negative)
        theta_neg = df_neg["theta"].values
        neg_ratio = df_neg["R_env"].values

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.axvspan(0.7, 1.4, color="#d9d9d9", alpha=0.35, zorder=0)
    # Emphasize baseband with markers + a light connecting line + SEM whiskers
    ax.errorbar(
        theta,
        ratio,
        yerr=ratio_sem,
        fmt="^",
        color="tab:purple",
        ecolor="tab:purple",
        elinewidth=0.8,
        capsize=2.0,
        label="Detuned + Kerr (baseband)",
    )
    ax.plot(theta, ratio, color="tab:purple", linewidth=0.8, alpha=0.55)
    if not np.all(np.isnan(env_ratio)):
        ax.plot(theta, env_ratio, linestyle=":", color="tab:green", linewidth=1.0, alpha=0.6, label="Periodogram ratio")
    if not np.all(np.isnan(hilbert_ratio)):
        ax.plot(theta, hilbert_ratio, linestyle="--", color="tab:olive", linewidth=1.0, alpha=0.6, label="Hilbert ratio")
    if surrogate_ratio is not None:
        ax.plot(theta_s, surrogate_ratio, marker="s", linestyle="--", color="tab:gray", label="Detuned linear control")
    if neg_ratio is not None:
        ax.plot(theta_neg, neg_ratio, marker="d", linestyle="-.", color="tab:pink", label="Detuned −0.1 + Kerr")

    ax.set_xlabel(r"$\Theta = \omega_1 \tau_B$")
    ax.set_ylabel(r"$R_\mathrm{env}$ relative to Markovian baseline")
    ax.set_title("Quantum equal-carrier sweep with Kerr + detuning")
    ax.grid(True, alpha=0.3)
    theta_min = theta.min()
    theta_max = theta.max()
    if theta_s is not None:
        theta_min = min(theta_min, theta_s.min())
        theta_max = max(theta_max, theta_s.max())
    if theta_neg is not None:
        theta_min = min(theta_min, theta_neg.min())
        theta_max = max(theta_max, theta_neg.max())
    ax.set_xlim(theta_min - 0.05, theta_max + 0.05)
    # Align x ticks to the evaluated Theta points for clarity
    ax.set_xticks(theta)
    values = [ratio]
    if surrogate_ratio is not None:
        values.append(surrogate_ratio)
    if neg_ratio is not None:
        values.append(neg_ratio)
    if not np.all(np.isnan(env_ratio)):
        values.append(env_ratio)
    if not np.all(np.isnan(hilbert_ratio)):
        values.append(hilbert_ratio)
    ymin = min(v.min() for v in values)
    ymax = max(v.max() for v in values)
    lower = max(0.95, ymin * 0.98)
    upper = ymax * 1.02
    ax.set_ylim(lower, upper)
    # Label the MR band explicitly
    ax.text(0.72, lower + 0.02 * (upper - lower), "MR band [0.7, 1.4]", color="#666666", fontsize=9)
    ax.legend(loc="upper left")

    # Annotate mechanism class near the detuned + Kerr curve
    peak_idx = int(np.nanargmax(ratio))
    ax.text(
        theta[peak_idx] + 0.02,
        ratio[peak_idx] + 0.01 * (upper - lower),
        "Class M",
        color="tab:purple",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # Equal-carrier inset showing |ΔJ|/J*
    rel_err = np.abs(df0.get("rel_j_err", pd.Series(dtype=float))).to_numpy()
    if rel_err.size and not np.all(np.isnan(rel_err)):
        rel_err = np.where(rel_err > 0, rel_err, 1e-12)
        inset = ax.inset_axes([0.63, 0.52, 0.32, 0.4])
        inset.semilogy(theta, rel_err, marker="o", color="tab:purple", linewidth=1.0)
        inset.axhline(1e-3, color="gray", linestyle="--", linewidth=0.9)
        inset.set_ylim(1e-12, 5e-3)
        inset.set_xlabel(r"$\Theta$", fontsize=8)
        inset.set_ylabel(r"$|\Delta J|/J^\star$", fontsize=8)
        inset.set_title("Equal-carrier gate", fontsize=8)
        inset.set_xticks(theta)
        inset.tick_params(axis="x", labelrotation=45)
        inset.tick_params(axis="both", labelsize=7)
        inset.text(
            0.02,
            0.82,
            r"$10^{-3}$",
            transform=inset.transAxes,
            fontsize=7,
            color="gray",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"✓ Saved figure to {args.output}")


if __name__ == "__main__":
    main()
