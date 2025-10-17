#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.metrics import baseband_power, narrowband_psd_power
from hierarchical_analysis import run_hierarchy, W1, W3


def main() -> None:
    p = argparse.ArgumentParser(description="Compare classical energy proxies: var(x1) vs band-limited around ω1")
    p.add_argument("--theta-list", type=float, nargs="*", default=[0.8, 1.0, 1.2])
    p.add_argument("--delays", type=float, nargs="*", default=[0.0, 0.5, 1.0])
    p.add_argument("--t-final", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--burn-frac", type=float, default=0.25)
    p.add_argument("--output", type=Path, default=Path("figures/figN_energy_proxy_compare.pdf"))
    args = p.parse_args()

    thetas = np.array(args.theta_list, dtype=float)
    delays = np.array(args.delays, dtype=float) * (1.0 / W1)

    E_var = []
    E_band = []
    labels = []
    for th in thetas:
        tau_B = float(th / W1)
        for d in delays:
            times, x1, x2, x3 = run_hierarchy(
                tau_B=tau_B,
                T=float(args.t_final),
                dt=float(args.dt),
                calibration_mode="equal_carrier",
                noise_param=1.0,
                burn_in_fraction=float(args.burn_frac),
            )
            fs = 1.0 / (times[1] - times[0])
            E_var.append(float(np.var(x1)))
            E_band.append(
                narrowband_psd_power(x1, fs_hz=fs, w3_rad_s=W1, rel_bandwidth=0.3, nperseg=4096, noverlap=2048)
            )
            labels.append(f"Θ={th:.2f}, d={d*W1:.2f}")

    E_var = np.asarray(E_var)
    E_band = np.asarray(E_band)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(E_var, E_band, color="tab:blue", edgecolor="k", linewidth=0.5)
    ax.plot([E_var.min(), E_var.max()], [E_var.min(), E_var.max()], color="#888888", linestyle=":")
    for i, lab in enumerate(labels):
        ax.annotate(lab, (E_var[i], E_band[i]), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xlabel("E_old = var(x1)")
    ax.set_ylabel("E_band = band-limited around ω1")
    ax.set_title("Classical energy proxy comparison")
    ax.grid(True, alpha=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"✓ Saved {args.output}")


if __name__ == "__main__":
    main()

