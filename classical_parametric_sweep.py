#!/usr/bin/env python3
"""Parametric coupling sweep to probe Class C behaviour."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

from hierarchical_analysis import run_hierarchy, ou_process_exact
from analysis.metrics import baseband_power, narrowband_psd_power
from analysis.metrics import hilbert_envelope_power
from analysis.surrogates import psd_matched_surrogate

from hierarchical_analysis import W1, W2, W3, G12


def compute_metrics(times: np.ndarray, signal: np.ndarray, fs: float, w3: float, lp_frac: float, nb_rel_bw: float, nperseg: int, noverlap: int) -> dict[str, float]:
    band = baseband_power(signal, fs, w3, lp_frac)
    narrow = narrowband_psd_power(signal, fs, w3, nb_rel_bw, nperseg, noverlap)
    env = hilbert_envelope_power(signal)
    return {"band": band, "narrow": narrow, "env": env}


def psd_nrmse(x: np.ndarray, y: np.ndarray, fs: float, nperseg: int, noverlap: int) -> float:
    from scipy.signal import welch

    freq_x, psd_x = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, psd_y = welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap)
    denom = np.sqrt(np.mean(psd_x**2))
    if denom == 0:
        return float("nan")
    diff = psd_x - psd_y
    return float(np.sqrt(np.mean(diff**2)) / denom)


def sweep(args: argparse.Namespace) -> list[dict[str, float]]:
    thetas = np.linspace(args.theta_min, args.theta_max, args.n_theta, dtype=float)
    seeds = list(range(args.n_seeds))
    modulation_freq = args.mod_omega * W1 if args.mod_units == "fraction" else args.mod_omega

    rows: list[dict[str, float]] = []
    baseline_value = None

    for theta in tqdm(thetas, desc="Theta"):
        tau_B = theta / W1
        seed_metrics_ou = []
        seed_metrics_psd = []
        psd_diffs: list[float] = []
        band_diff: list[float] = []
        for seed in seeds:
            noise = ou_process_exact(
                T=args.duration,
                dt=args.dt,
                tau_B=tau_B,
                target_variance=args.noise_var,
                seed=seed,
            )
            g12_mod = lambda t, base=G12: base * (1.0 + args.mod_amp * math.cos(modulation_freq * t))
            times, _, _, x3 = run_hierarchy(
                tau_B=tau_B,
                T=args.duration,
                dt=args.dt,
                calibration_mode="equal_variance",
                noise_param=args.noise_var,
                burn_in_fraction=args.burn_frac,
                noise_override=noise,
                g12_modulation=g12_mod,
            )
            burn_idx = int(args.burn_frac * x3.size)
            t_post = times[burn_idx:]
            x_post = x3[burn_idx:]
            fs = 1.0 / args.dt
            metrics = compute_metrics(t_post, x_post, fs, W3, args.lp_frac, args.nb_rel_bw, args.welch_nperseg, args.welch_noverlap)
            seed_metrics_ou.append(metrics)

            surrogate_noise = psd_matched_surrogate(noise, rng=np.random.default_rng(seed + 10_000))
            _, _, _, x3_surr = run_hierarchy(
                tau_B=tau_B,
                T=args.duration,
                dt=args.dt,
                calibration_mode="equal_variance",
                noise_param=args.noise_var,
                burn_in_fraction=args.burn_frac,
                noise_override=surrogate_noise,
                g12_modulation=g12_mod,
            )
            x_surr = x3_surr[burn_idx:]
            metrics_surr = compute_metrics(t_post, x_surr, fs, W3, args.lp_frac, args.nb_rel_bw, args.welch_nperseg, args.welch_noverlap)
            seed_metrics_psd.append(metrics_surr)
            psd_diffs.append(
                psd_nrmse(
                    x_post,
                    x_surr,
                    fs,
                    args.welch_nperseg,
                    args.welch_noverlap,
                )
            )
            band_diff.append(metrics["band"] - metrics_surr["band"])

        def aggregate(metric_list: Sequence[dict[str, float]], key: str) -> tuple[float, float]:
            values = np.array([m[key] for m in metric_list], dtype=float)
            return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else (values[0], 0.0)

        mean_band, sem_band = aggregate(seed_metrics_ou, "band")
        mean_band_surr, sem_band_surr = aggregate(seed_metrics_psd, "band")
        mean_env, sem_env = aggregate(seed_metrics_ou, "env")
        mean_env_surr, sem_env_surr = aggregate(seed_metrics_psd, "env")

        if baseline_value is None:
            baseline_value = mean_band
        ratio_band = mean_band / baseline_value if baseline_value not in (None, 0.0) else float("nan")
        ratio_band_surr = mean_band_surr / baseline_value if baseline_value not in (None, 0.0) else float("nan")

        rows.append(
            {
                "theta": theta,
                "tau_B": tau_B,
                "band_mean": mean_band,
                "band_sem": sem_band,
                "band_ratio": ratio_band,
                "band_mean_surrogate": mean_band_surr,
                "band_sem_surrogate": sem_band_surr,
                "band_ratio_surrogate": ratio_band_surr,
                "env_mean": mean_env,
                "env_sem": sem_env,
                "env_mean_surrogate": mean_env_surr,
                "env_sem_surrogate": sem_env_surr,
                "psd_nrmse": float(np.nanmean(psd_diffs)),
                "cohen_dz": (np.mean(band_diff) / np.std(band_diff, ddof=1)) if len(band_diff) > 1 else float("nan"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical parametric modulation sweep")
    parser.add_argument("--output", type=Path, default=Path("results/classical_parametric.csv"))
    parser.add_argument("--theta-min", type=float, default=0.6)
    parser.add_argument("--theta-max", type=float, default=1.6)
    parser.add_argument("--n-theta", type=int, default=7)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--burn-frac", type=float, default=0.25)
    parser.add_argument("--noise-var", type=float, default=1.0)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--lp-frac", type=float, default=0.2)
    parser.add_argument("--nb-rel-bw", type=float, default=0.3)
    parser.add_argument("--welch-nperseg", type=int, default=8192)
    parser.add_argument("--welch-noverlap", type=int, default=4096)
    parser.add_argument("--mod-amp", type=float, default=0.15)
    parser.add_argument("--mod-omega", type=float, default=1.0, help="Modulation frequency multiplier or absolute (see mod-units)")
    parser.add_argument("--mod-units", choices=["fraction", "absolute"], default="fraction")
    args = parser.parse_args()

    rows = sweep(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"✓ Saved results to {args.output}")


if __name__ == "__main__":
    main()
