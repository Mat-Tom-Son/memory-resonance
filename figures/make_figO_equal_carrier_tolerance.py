#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_concat(csvs: list[Path]) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in csvs if p.exists()]
    if not dfs:
        raise FileNotFoundError("No CSVs found for tolerance sweep plot")
    df = pd.concat(dfs, ignore_index=True)
    # Keep only pseudomode model rows if 'model' column exists
    if 'model' in df.columns:
        df = df[df['model'] == 'pseudomode']
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Equal-carrier tolerance sweep overlay")
    p.add_argument("--csvs", type=Path, nargs="+", default=[
        Path("results/adaptive/kerr08_det12_theta_grid7.csv"),
        Path("results/adaptive/kerr08_det12_theta_grid7_rep2.csv"),
        Path("results/adaptive/kerr08_det12_theta_grid7_rep3.csv"),
    ])
    p.add_argument("--tols", type=float, nargs="*", default=[5e-4, 1e-3, 2e-3])
    p.add_argument("--output", type=Path, default=Path("figures/figO_equal_carrier_tolerance.pdf"))
    args = p.parse_args()

    df = load_concat(list(args.csvs))
    theta = np.sort(df['theta'].unique())

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for tol, ls, col in zip(args.tols, ['-', '--', ':'], ['tab:blue', 'tab:green', 'tab:orange']):
        mask = (df['rel_j_err'].abs() <= tol) if 'rel_j_err' in df.columns else np.ones(len(df), dtype=bool)
        dft = df[mask]
        # Average across repeats for each theta under this tol mask
        Rmean = []
        for th in theta:
            vals = dft.loc[dft['theta'] == th, 'R_env'].to_numpy()
            Rmean.append(np.nanmean(vals) if vals.size else np.nan)
        Rmean = np.asarray(Rmean, dtype=float)
        ax.plot(theta, Rmean, linestyle=ls, color=col, marker='o', label=fr"tol = {tol:.1e}")

    ax.set_xlabel(r"$\Theta$")
    ax.set_ylabel(r"$R_{env}$")
    ax.set_title("Equal-carrier tolerance robustness")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"✓ Saved {args.output}")


if __name__ == "__main__":
    main()

