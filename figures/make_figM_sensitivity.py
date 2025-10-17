#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quant.gaussian_linear import build_model, psd as gl_psd
from quantum_models import W1, W2, W3, G12, G23, GAMMA1, GAMMA2, GAMMA3


def main() -> None:
    p = argparse.ArgumentParser(description="Operational sensitivity mask δR/δS_ξ proxy")
    p.add_argument("--output", type=Path, default=Path("figures/figM_sensitivity.pdf"))
    args = p.parse_args()

    # Build linear Gaussian model and compute |H|^2 as a proxy
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
    # Zero out a narrow band around ω1 to reflect equal-carrier invariance
    mask = np.ones_like(omega, dtype=bool)
    band_lo = W1 * 0.96
    band_hi = W1 * 1.04
    mask &= ~((omega >= band_lo) & (omega <= band_hi))
    sens = H2.copy()
    sens[~mask] = 0.0
    # Normalize for display
    sens_n = sens / max(1e-18, sens.max())

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(omega, sens_n, color="tab:red", linewidth=1.4, label=r"Sensitivity proxy $|H(\omega)|^2$ (notch at $\omega_1$)")
    ax.axvline(W1, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("Normalized sensitivity (a.u.)")
    ax.set_title("Operational gradient mask δR/δS_ξ(ω) proxy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"✓ Saved {args.output}")


if __name__ == "__main__":
    main()

