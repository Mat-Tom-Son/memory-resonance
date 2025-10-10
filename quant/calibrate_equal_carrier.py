"""Utility to precompute equal-carrier couplings across Θ values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from analysis.config_loader import load_config_with_hash
from quantum_models import (
    GAMMA1,
    W1,
    baseline_pseudomode_coupling,
    calibrate_equal_carrier_g,
    pseudomode_lorentzian_j,
)


def parse_theta_list(theta_str: str) -> List[float]:
    return [float(token.strip()) for token in theta_str.split(",") if token.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute equal-carrier couplings")
    parser.add_argument(
        "--thetas",
        type=str,
        required=True,
        help="Comma-separated Θ values to calibrate.",
    )
    parser.add_argument(
        "--baseline_theta",
        type=float,
        default=1.0,
        help="Θ value used as baseline reference (default: 1.0).",
    )
    parser.add_argument(
        "--tol_Jw1",
        type=float,
        default=0.02,
        help="Relative tolerance for matching spectral weight at ω₁.",
    )
    parser.add_argument(
        "--omega_c_scale",
        type=float,
        default=1.0,
        help="Scale factor for pseudomode frequency relative to ω₁.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for the calibration map.",
    )
    args = parser.parse_args()

    theta_values = parse_theta_list(args.thetas)
    if not theta_values:
        raise ValueError("No Θ values provided for calibration.")

    config, config_hash = load_config_with_hash()

    omega_c = args.omega_c_scale * W1
    delta = W1 - omega_c

    baseline_tau = args.baseline_theta / W1
    if baseline_tau <= 0:
        raise ValueError("Baseline Θ must correspond to positive τ_B.")
    baseline_kappa = 1.0 / baseline_tau
    baseline_g = baseline_pseudomode_coupling(baseline_tau, "equal_carrier", W1, GAMMA1)
    target_j = pseudomode_lorentzian_j(baseline_g, baseline_kappa, delta)

    map_entries = {}
    for theta in theta_values:
        tau_B = theta / W1
        if tau_B <= 0:
            raise ValueError(f"Θ={theta} yields non-positive τ_B.")
        kappa = 1.0 / tau_B
        g_cal, _, j_est, rel_err = calibrate_equal_carrier_g(
            kappa=kappa,
            baseline_kappa=baseline_kappa,
            baseline_g=baseline_g,
            delta=delta,
            tolerance=args.tol_Jw1,
        )
        map_entries[f"{theta:.6f}"] = {
            "theta": theta,
            "tau_B": tau_B,
            "kappa": kappa,
            "g": g_cal,
            "Jw1_target": target_j,
            "Jw1_est": j_est,
            "rel_Jw1_err": rel_err,
            "delta": delta,
        }

    output = {
        "config_hash": config_hash,
        "baseline": {
            "theta": args.baseline_theta,
            "tau_B": baseline_tau,
            "kappa": baseline_kappa,
            "g": baseline_g,
            "Jw1_target": target_j,
            "delta": delta,
        },
        "entries": map_entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    print(f"✓ Equal-carrier map saved to {args.output}")


if __name__ == "__main__":
    main()
