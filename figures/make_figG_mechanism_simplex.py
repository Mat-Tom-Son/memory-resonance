#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.mechanism_simplex import compute_simplex


def barycentric_to_xy(bary: tuple[float, float, float]) -> tuple[float, float]:
    # Equilateral triangle with vertices at (0,0)->S, (1,0)->C, (0.5, sqrt(3)/2)->M
    S, C, M = bary
    vS = np.array([0.0, 0.0])
    vC = np.array([1.0, 0.0])
    vM = np.array([0.5, np.sqrt(3.0) / 2.0])
    p = S * vS + C * vC + M * vM
    return float(p[0]), float(p[1])


def draw_simplex(ax: plt.Axes) -> None:
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0], [0.0, 0.0]])
    ax.plot(verts[:, 0], verts[:, 1], color="#333333")
    ax.text(-0.03, -0.04, "S (spectral)")
    ax.text(1.01, -0.04, "C (coherence)")
    ax.text(0.48, np.sqrt(3.0) / 2.0 + 0.03, "M (memory)")
    ax.set_aspect("equal")
    ax.set_axis_off()


def main() -> None:
    p = argparse.ArgumentParser(description="Mechanism simplex plot")
    p.add_argument("--classical-csv", type=Path, default=Path("results/theta_sweep_today.csv"))
    p.add_argument("--classical-parametric-csv", type=Path, default=Path("results/adaptive/classC_fraction_attempt.csv"))
    p.add_argument("--quantum-equal-carrier-csv", type=Path, default=Path("results/adaptive/kerr08_det12_theta_grid7.csv"))
    p.add_argument("--output", type=Path, default=Path("figures/figG_mechanism_simplex.pdf"))
    args = p.parse_args()

    res = compute_simplex(
        classical_csv=args.classical_csv,
        classical_parametric_csv=args.classical_parametric_csv,
        quantum_equal_carrier_csv=args.quantum_equal_carrier_csv,
    )
    bary = res.barycentric()
    x, y = barycentric_to_xy(bary)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    draw_simplex(ax)
    # Point and uncertainty halo
    ax.scatter([x], [y], color="tab:purple", s=60, label="This work (adaptive)")
    # Simple uncertainty radius from mean SEM (visual only)
    mean_sem = float(np.mean([res.S_sem, res.C_sem, res.M_sem]))
    radius = 0.06 + 0.10 * mean_sem  # small visual halo
    circ = plt.Circle((x, y), radius, color="tab:purple", alpha=0.12, linewidth=0)
    ax.add_patch(circ)
    ax.legend(loc="upper right")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"✓ Saved {args.output}")


if __name__ == "__main__":
    main()
