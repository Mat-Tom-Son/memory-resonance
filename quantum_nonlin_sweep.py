#!/usr/bin/env python3
"""
Equal-carrier sweep for nonlinear/ detuned pseudomode variants.
Produces a CSV with band-power ratios etc so we can hunt for Class M/C signatures.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable
import time

import numpy as np
from tqdm import tqdm

from analysis.metrics import (
    baseband_power,
    narrowband_psd_power,
    hilbert_envelope_power,
)
from analysis.metrics import butterworth_lpf_gain_sq  # imported to reuse configs if needed
from quantum_models import (
    W1,
    W3,
    G12,
    G23,
    GAMMA1,
    GAMMA2,
    GAMMA3,
    run_markovian_model,
    run_pseudomode_model,
    baseline_pseudomode_coupling,
    calibrate_equal_carrier_g,
    pseudomode_lorentzian_j,
)


def envelope_power(times: np.ndarray, signal: np.ndarray, band: tuple[float, float]) -> float:
    from scipy.signal import periodogram

    dt = float(times[1] - times[0])
    freq, psd = periodogram(signal, fs=1.0 / dt)
    mask = (freq >= band[0]) & (freq <= band[1])
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freq[mask]))


def rolling_mean(series: np.ndarray) -> float:
    if series.size == 0:
        return float("nan")
    return float(np.mean(series))


def prepare_theta_grid(args: argparse.Namespace) -> np.ndarray:
    if args.theta_list:
        return np.array(sorted(set(args.theta_list)), dtype=float)
    theta_min = args.theta_min if args.theta_min is not None else 0.5
    theta_max = args.theta_max if args.theta_max is not None else 1.6
    n_points = args.n_theta if args.n_theta is not None else 9
    if theta_min <= 0 or theta_max <= 0:
        raise ValueError("theta grid must be positive")
    if args.theta_spacing == "linear":
        return np.linspace(theta_min, theta_max, n_points, dtype=float)
    return np.geomspace(theta_min, theta_max, n_points, dtype=float)


def sweep(args: argparse.Namespace) -> list[dict[str, float]]:
    thetas = prepare_theta_grid(args)
    t_final = float(args.t_final)
    n_time = int(args.n_time)
    times = np.linspace(0.0, t_final, n_time, dtype=float)
    burn_idx = int(args.burn_frac * n_time)
    times_post = times[burn_idx:]
    omega_c = W1 * (1.0 + args.detune_frac)
    delta_local = W1 - omega_c

    # Baseline equal-carrier reference (Θ=1 point)
    tau_ref = 1.0 / W1
    baseline_kappa = 1.0 / tau_ref
    baseline_g = baseline_pseudomode_coupling(
        tau_ref,
        "equal_carrier",
        W1,
        GAMMA1,
    )
    target_j = pseudomode_lorentzian_j(baseline_g, baseline_kappa, delta_local)

    # Markovian baseline (no pseudomode) for ratios
    markov_args = dict(
        N=args.n_hierarchy,
        times=times,
        excitation_amp=args.excitation_amp,
        calibration_mode=None,
        nbar=args.nbar,
        a1_init_alpha=args.a1_init_alpha,
    )
    t_markov, x3_markov, n3_markov = run_markovian_model(**markov_args)
    x3m = x3_markov[burn_idx:]
    t_m = t_markov[burn_idx:]
    dt = t_m[1] - t_m[0]
    fs = 1.0 / dt
    base_band = baseband_power(x3m, fs, W3, args.lp_frac)
    base_env = envelope_power(t_m, x3m, args.env_band)
    base_hilbert = hilbert_envelope_power(x3m)
    base_narrow = narrowband_psd_power(
        x3m,
        fs,
        W3,
        args.nb_rel_bw,
        args.welch_nperseg,
        args.welch_noverlap,
    )
    base_n1 = rolling_mean(n3_markov[burn_idx:])

    rows: list[dict[str, float]] = []
    start_time = time.time()
    patience_counter = 0
    for theta in tqdm(thetas, desc="Theta sweep"):
        tau_B = float(theta / W1)
        kappa = 1.0 / tau_B
        g_override, target_j_local, j_est, rel_err = calibrate_equal_carrier_g(
            kappa=kappa,
            baseline_kappa=baseline_kappa,
            baseline_g=baseline_g,
            delta=delta_local,
            tolerance=args.equal_carrier_tol,
        )
        model_kwargs = dict(
            tau_B=tau_B,
            N=args.n_hierarchy,
            N_pseudo=args.n_pseudo,
            times=times,
            excitation_amp=args.excitation_amp,
            calibration_mode="equal_carrier",
            g_override=g_override,
            omega_c=omega_c,
            kerr_fast=args.kerr_fast,
            return_n1=True,
            nbar=args.nbar,
            a1_init_alpha=args.a1_init_alpha,
        )
        sim = run_pseudomode_model(**model_kwargs)
        times_p = sim[0]
        x3_p = sim[1]
        n3_p = sim[3]
        t_post = times_p[burn_idx:]
        x_post = x3_p[burn_idx:]
        dt_p = t_post[1] - t_post[0]
        fs_p = 1.0 / dt_p
        band_power = baseband_power(x_post, fs_p, W3, args.lp_frac)
        env_power = envelope_power(t_post, x_post, args.env_band)
        hilbert_power = hilbert_envelope_power(x_post)
        narrow_power = narrowband_psd_power(
            x_post,
            fs_p,
            W3,
            args.nb_rel_bw,
            args.welch_nperseg,
            args.welch_noverlap,
        )
        n1_avg = rolling_mean(n3_p[burn_idx:])
        n1_ratio = n1_avg / base_n1 if base_n1 > 0 else float("nan")

        rows.append(
            {
                "theta": theta,
                "tau_B": tau_B,
                "R_env": band_power / base_band if base_band else float("nan"),
                "R_narrow": narrow_power / base_narrow if base_narrow else float("nan"),
                "R_env_periodogram": env_power / base_env if base_env else float("nan"),
                "R_env_hilbert": hilbert_power / base_hilbert if base_hilbert else float("nan"),
                "band_power": band_power,
                "base_band": base_band,
                "narrow_power": narrow_power,
                "env_power": env_power,
                "hilbert_power": hilbert_power,
                "base_narrow": base_narrow,
                "base_env": base_env,
                "base_hilbert": base_hilbert,
                "g_used": g_override,
                "target_j": target_j_local,
                "j_est": j_est,
                "rel_j_err": rel_err,
                "omega_c": omega_c,
                "kerr_fast": args.kerr_fast,
                "n1_mean": n1_avg,
                "n1_base": base_n1,
                "n1_ratio": n1_ratio,
                "excitation_amp": args.excitation_amp,
                "n_hierarchy": args.n_hierarchy,
                "n_pseudo": args.n_pseudo,
                "burn_frac": args.burn_frac,
                "detune_frac": args.detune_frac,
            }
        )

        # Early-stop logic: optional runtime and efficacy gates
        if getattr(args, "early_stop", False):
            scanned = len(rows)
            best_r = max([r.get("R_env", float("nan")) for r in rows])
            if scanned >= getattr(args, "early_min_theta", 3):
                if not np.isfinite(best_r) or best_r < getattr(args, "early_threshold", 1.08):
                    patience_counter += 1
                else:
                    patience_counter = 0
                if patience_counter > getattr(args, "early_patience", 1):
                    print(
                        f"Early stop: best R_env={best_r:.3f} below threshold "
                        f"{getattr(args, 'early_threshold', 1.08):.3f} after {scanned} points."
                    )
                    break
            max_rt = getattr(args, "max_runtime_s", None)
            if max_rt is not None and (time.time() - start_time) > float(max_rt):
                print(f"Early stop: exceeded runtime budget {max_rt}s after {scanned} points.")
                break
    return rows


def write_csv(rows: Iterable[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"✓ Saved sweep to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Equal-carrier sweep with nonlinear pseudomode")
    parser.add_argument("--output", type=Path, default=Path("results/quantum_nonlin/sweep.csv"))
    parser.add_argument("--theta-list", type=float, nargs="*")
    parser.add_argument("--theta-min", type=float)
    parser.add_argument("--theta-max", type=float)
    parser.add_argument("--n-theta", type=int)
    parser.add_argument("--theta-spacing", choices=["linear", "log"], default="linear")
    parser.add_argument("--detune-frac", type=float, default=0.0, help="Pseudomode detuning fraction (omega_c = W1*(1+delta))")
    parser.add_argument("--kerr-fast", type=float, default=0.0, help="Kerr nonlinearity strength on fast mode")
    parser.add_argument("--t-final", type=float, default=2.5)
    parser.add_argument("--n-time", type=int, default=3000)
    parser.add_argument("--burn-frac", type=float, default=0.25)
    parser.add_argument("--excitation-amp", type=float, default=0.05)
    parser.add_argument("--n-hierarchy", type=int, default=6)
    parser.add_argument("--n-pseudo", type=int, default=4)
    parser.add_argument("--lp-frac", type=float, default=0.2)
    parser.add_argument("--nb-rel-bw", type=float, default=0.3)
    parser.add_argument("--welch-nperseg", type=int, default=512)
    parser.add_argument("--welch-noverlap", type=int, default=256)
    parser.add_argument("--env-band", type=float, nargs=2, default=(0.5, 5.0))
    parser.add_argument("--equal-carrier-tol", type=float, default=0.02)
    parser.add_argument("--nbar", type=float, default=0.0)
    parser.add_argument("--a1-init-alpha", type=float, default=0.0)
    parser.add_argument("--seed", type=int, help="Not used but kept for interface symmetry")
    # Early-stop controls
    parser.add_argument("--early-stop", action="store_true", help="Enable early termination if lift is not promising")
    parser.add_argument("--early-threshold", type=float, default=1.08, help="Minimum R_env target to keep scanning")
    parser.add_argument("--early-min-theta", type=int, default=3, help="Minimum Θ points before early-stop checks")
    parser.add_argument("--early-patience", type=int, default=1, help="Allowed consecutive misses below threshold")
    parser.add_argument("--max-runtime-s", type=float, help="Optional wall-clock budget (seconds)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = sweep(args)
    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
