#!/usr/bin/env python3
"""Cross-domain collapse plot normalised by Θ."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_classical_s(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    mask = (df["model"] == "classical") & (df["condition"] == "ou") & (df["metric"] == "baseband_power")
    data = df.loc[mask, ["theta", "value"]]
    grouped = data.groupby("theta").mean().reset_index()
    theta = grouped["theta"].values
    values = grouped["value"].values
    return theta, values / values.min()


def load_classical_c(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    theta = df["theta"].values
    ratio = df["band_ratio"].values
    sem = df["band_sem"].values
    baseline = df.loc[df.index[0], "band_mean"]
    sem_ratio = sem / baseline
    return theta, ratio, df["band_ratio_surrogate"].values, sem_ratio


def load_quantum(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    theta = df["theta"].values
    ratio = df["R_env"].values
    return theta, ratio / ratio.min()


def load_quantum_null(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    theta = df["theta"].values
    ratio = df["R_env"].values
    return theta, ratio / ratio.min()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical-s", type=Path, default=Path("results/theta_sweep_today.csv"))
    parser.add_argument("--classical-c", type=Path, default=Path("results/classical_parametric_mod03.csv"))
    parser.add_argument("--quantum-positive", type=Path, default=Path("results/quantum_nonlin/kerr02_tol001.csv"))
    parser.add_argument("--quantum-positive-neg", type=Path, default=Path("results/quantum_nonlin/kerr02_tol001_neg.csv"))
    parser.add_argument("--quantum-null", type=Path, default=Path("results/quantum_eqheat_sweep_for_figB.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures/figE_collapse.pdf"))
    args = parser.parse_args()

    theta_s, ratio_s = load_classical_s(args.classical_s)
    theta_c, ratio_c, ratio_c_surr, sem_c = load_classical_c(args.classical_c)
    theta_q, ratio_q = load_quantum(args.quantum_positive)
    theta_q_neg, ratio_q_neg = load_quantum(args.quantum_positive_neg) if args.quantum_positive_neg.exists() else (None, None)
    theta_null, ratio_null = load_quantum_null(args.quantum_null)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axvspan(0.7, 1.4, color="#d9d9d9", alpha=0.35, zorder=0)

    ax.plot(theta_s, ratio_s, marker="o", color="tab:blue", label="Class S (OU baseline)")
    ax.errorbar(theta_c, ratio_c, yerr=sem_c, marker="s", color="tab:orange", label="Class C (parametric)")
    ax.plot(theta_q, ratio_q, marker="^", color="tab:purple", label="Quantum detune+Kerr (δ=+0.1)")
    if theta_q_neg is not None:
        ax.plot(theta_q_neg, ratio_q_neg, marker="v", linestyle="--", color="tab:pink", label="Quantum detune+Kerr (δ=-0.1)")
    ax.plot(theta_null, ratio_null, linestyle="--", color="tab:gray", label="Quantum equal-carrier null")

    ax.set_xlabel(r"$\Theta = \omega_1 \tau_B$")
    ax.set_ylabel(r"$G(\Theta)$ (normalised envelope)")
    ax.set_ylim(0.95, max(ratio_q.max(), ratio_s.max(), ratio_c.max()) * 1.1)
    ax.set_xlim(min(theta_s.min(), theta_c.min(), theta_q.min()) - 0.05,
                max(theta_s.max(), theta_c.max(), theta_q.max()) + 0.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title("Cross-domain collapse onto the MRC band")

    def annotate_class(theta: np.ndarray, values: np.ndarray, text: str, color: str, dy: float = 0.015) -> None:
        if theta.size == 0 or values.size == 0:
            return
        idx = int(np.nanargmin(np.abs(theta - 1.0)))
        ax.text(
            theta[idx] + 0.03,
            values[idx] * (1.0 + dy),
            text,
            color=color,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    annotate_class(theta_s, ratio_s, "Class S", "tab:blue")
    annotate_class(theta_c, ratio_c, "Class C", "tab:orange", dy=0.03)
    annotate_class(theta_q, ratio_q, "Class M", "tab:purple")
    annotate_class(theta_null, ratio_null, "Class M (null)", "tab:gray", dy=0.01)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"✓ Saved collapse figure to {args.output}")


if __name__ == "__main__":
    main()
