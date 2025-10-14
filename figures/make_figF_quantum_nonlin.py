#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum equal-carrier sweep with detuning + Kerr")
    parser.add_argument("--csv", type=Path, default=Path("results/quantum_nonlin/kerr02_tol001.csv"))
    parser.add_argument("--csv-surrogate", type=Path, default=Path("results/quantum_nonlin/kerr00_tol001.csv"))
    parser.add_argument("--csv-negative", type=Path, default=Path("results/quantum_nonlin/kerr02_tol001_neg.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures/figF_quantum_nonlin.pdf"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    theta = df["theta"].values
    ratio = df["R_env"].values
    env_ratio = df.get("R_env_periodogram")
    env_ratio = env_ratio.to_numpy() if env_ratio is not None else np.full_like(ratio, np.nan)
    hilbert_ratio = df.get("R_env_hilbert")
    hilbert_ratio = hilbert_ratio.to_numpy() if hilbert_ratio is not None else np.full_like(ratio, np.nan)

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
    ax.fill_between([0.7, 1.4], 0.95, 1.15, color="lightgray", alpha=0.2, label="MR band")
    ax.plot(theta, ratio, marker="^", color="tab:purple", label="Detuned + Kerr (baseband)")
    if not np.all(np.isnan(env_ratio)):
        ax.plot(theta, env_ratio, linestyle=":", color="tab:green", label="Periodogram ratio")
    if not np.all(np.isnan(hilbert_ratio)):
        ax.plot(theta, hilbert_ratio, linestyle="--", color="tab:olive", label="Hilbert ratio")
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
    ax.set_ylim(ymin * 0.95, ymax * 1.05)
    ax.legend(loc="upper left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"✓ Saved figure to {args.output}")


if __name__ == "__main__":
    main()
