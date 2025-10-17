#!/usr/bin/env python3
"""
Adaptive battery to produce stronger Class C and Class M lifts with early-stop.

Runs a short Class C parametric sweep and a short Class M equal-carrier
nonlinear sweep, with early termination if the effect is not promising.

Targets a total runtime of a few minutes on a laptop; adjust per flags.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np

import classical_parametric_sweep as C
import quantum_nonlin_sweep as Q


def summarize_class_c(rows: list[dict]) -> Tuple[float, float]:
    """Return (max_psd_nrmse, max_gap) across thetas."""
    if not rows:
        return 0.0, 0.0
    psd_vals = []
    gaps = []
    for r in rows:
        psd_vals.append(float(r.get("psd_nrmse", 0.0)))
        br = float(r.get("band_ratio", np.nan))
        brs = float(r.get("band_ratio_surrogate", np.nan))
        gap = br - brs if np.isfinite(br) and np.isfinite(brs) else 0.0
        gaps.append(gap)
    return float(np.nanmax(psd_vals)), float(np.nanmax(gaps))


def summarize_class_m(rows: list[dict]) -> Tuple[float, float]:
    """Return (max_R_env, max_rel_j_err) across thetas."""
    if not rows:
        return 1.0, np.inf
    r_vals = []
    jerrs = []
    for r in rows:
        r_vals.append(float(r.get("R_env", np.nan)))
        jerrs.append(float(r.get("rel_j_err", np.nan)))
    r_max = np.nanmax(r_vals) if len(r_vals) else float("nan")
    jerr_max = np.nanmax(jerrs) if len(jerrs) else float("nan")
    return float(r_max), float(jerr_max)


def run_class_c(output_dir: Path, budget_s: float) -> Path:
    out1 = output_dir / "classC_attempt1.csv"
    args1 = SimpleNamespace(
        output=out1,
        theta_min=0.7,
        theta_max=1.5,
        n_theta=7,
        duration=25.0,
        dt=1e-3,
        burn_frac=0.25,
        noise_var=1.0,
        n_seeds=4,
        lp_frac=0.2,
        nb_rel_bw=0.3,
        welch_nperseg=4096,
        welch_noverlap=2048,
        mod_amp=0.35,
        mod_omega=1.0,
        mod_units="absolute",
        early_stop=True,
        early_psd_threshold=0.8,
        early_gap_threshold=0.2,
        early_min_theta=3,
        early_patience=1,
        max_runtime_s=min(budget_s * 0.45, budget_s - 15.0),
    )
    rows1 = C.sweep(args1)
    out1.parent.mkdir(parents=True, exist_ok=True)
    with out1.open("w", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=list(rows1[0].keys()) if rows1 else [])
        if rows1:
            writer.writeheader()
            writer.writerows(rows1)
    psd1, gap1 = summarize_class_c(rows1)
    print(f"Class C attempt1: max PSD-NRMSE={psd1:.2f}, max gap={gap1:.2f}")

    if psd1 >= 0.8 or gap1 >= 0.2:
        return out1

    # Escalate modulation depth for a second attempt
    out2 = output_dir / "classC_attempt2.csv"
    args2 = args1
    args2.output = out2
    args2.mod_amp = 0.45
    args2.max_runtime_s = max(30.0, budget_s * 0.45)
    rows2 = C.sweep(args2)
    with out2.open("w", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=list(rows2[0].keys()) if rows2 else [])
        if rows2:
            writer.writeheader()
            writer.writerows(rows2)
    psd2, gap2 = summarize_class_c(rows2)
    print(f"Class C attempt2: max PSD-NRMSE={psd2:.2f}, max gap={gap2:.2f}")
    return out2


def run_class_m(output_dir: Path, budget_s: float) -> Path:
    out1 = output_dir / "classM_attempt1.csv"
    args1 = SimpleNamespace(
        output=out1,
        theta_list=[0.9, 1.0, 1.1, 1.2],
        theta_min=None,
        theta_max=None,
        n_theta=None,
        theta_spacing="linear",
        detune_frac=0.12,
        kerr_fast=0.05,
        t_final=3.0,
        n_time=2400,
        burn_frac=0.4,
        excitation_amp=0.06,
        n_hierarchy=6,
        n_pseudo=5,
        lp_frac=0.2,
        nb_rel_bw=0.3,
        welch_nperseg=512,
        welch_noverlap=256,
        env_band=(0.5, 5.0),
        equal_carrier_tol=0.001,
        nbar=0.0,
        a1_init_alpha=0.2,
        seed=None,
        early_stop=True,
        early_threshold=1.08,
        early_min_theta=3,
        early_patience=1,
        max_runtime_s=min(budget_s * 0.45, budget_s - 15.0),
    )
    rows1 = Q.sweep(args1)
    Q.write_csv(rows1, out1)
    r1, jerr1 = summarize_class_m(rows1)
    print(f"Class M attempt1: max R_env={r1:.3f}, max |ΔJ|/J*={jerr1:.3e}")

    if np.isfinite(r1) and r1 >= 1.08 and (not np.isfinite(jerr1) or jerr1 <= 2e-3):
        return out1

    # Escalate kerr strength for a second attempt
    out2 = output_dir / "classM_attempt2.csv"
    args2 = args1
    args2.output = out2
    args2.kerr_fast = 0.07
    args2.max_runtime_s = max(30.0, budget_s * 0.45)
    rows2 = Q.sweep(args2)
    Q.write_csv(rows2, out2)
    r2, jerr2 = summarize_class_m(rows2)
    print(f"Class M attempt2: max R_env={r2:.3f}, max |ΔJ|/J*={jerr2:.3e}")
    return out2


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive battery for stronger Class C/M lifts with early-stop")
    parser.add_argument("--output-dir", type=Path, default=Path("results/adaptive"))
    parser.add_argument("--budget-min", type=float, default=8.0, help="Overall soft runtime budget in minutes")
    parser.add_argument("--skip-class-c", action="store_true")
    parser.add_argument("--skip-class-m", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    budget_s = max(60.0, float(args.budget_min) * 60.0)

    t0 = time.time()
    outputs = []
    if not args.skip_class_c:
        print("\n=== Running Class C (parametric) with early-stop ===")
        p = run_class_c(args.output_dir, budget_s)
        outputs.append(p)

    if not args.skip_class_m:
        print("\n=== Running Class M (equal-carrier, detuned+Kerr) with early-stop ===")
        p = run_class_m(args.output_dir, budget_s)
        outputs.append(p)

    dt = time.time() - t0
    print("\nSummary:")
    for p in outputs:
        print(f"  - {p}")
    print(f"Total wall time: {dt/60.0:.2f} min")


if __name__ == "__main__":
    main()

