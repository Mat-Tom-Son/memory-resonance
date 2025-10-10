"""Classical control experiment comparing OU drive to PSD-matched surrogate."""

from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

from analysis.config_loader import load_config_with_hash
from analysis.metrics import baseband_power, narrowband_psd_power
from analysis.stats import bootstrap_ci, cohen_d, paired_cohen_d, tost_equivalence_paired
from analysis.surrogates import psd_matched_surrogate
from hierarchical_analysis import ou_process_exact, run_hierarchy
from quantum_models import W1


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _resolve_welch_params(length: int, requested_nperseg: int, requested_noverlap: int) -> tuple[int, int]:
    """Choose a Welch segment length that yields at least 16 segments."""
    length = max(int(length), 1)
    target = max(length // 16, 1)
    power = max(int(np.floor(np.log2(target))), 4)
    nperseg = 2 ** power
    nperseg = min(nperseg, length)
    if requested_nperseg > 0:
        nperseg = min(nperseg, int(requested_nperseg))
    nperseg = max(16, nperseg)
    noverlap = min(int(requested_noverlap), nperseg // 2)
    return nperseg, noverlap


def run_classical_chain(
    config: dict,
    config_hash: str,
    output_dir: Path,
    calibration_mode: str = "equal_variance",
    noise_param: float = 1.0,
    base_seed: int = 1337,
    theta_values: list[float] | None = None,
    seeds_peak: int = 20,
    seeds_flank: int = 8,
    t_final_override: float | None = None,
    burn_frac_override: float | None = None,
    welch_nperseg_override: int | None = None,
    welch_noverlap_override: int | None = None,
    nbw_rel_override: float | None = None,
    psd_norm: str = "onesided",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    print(f"[RUN] run_id={run_id} output_dir={output_dir}")

    fs_hz = float(config["fs_hz"])
    dt = 1.0 / fs_hz
    w3_hz = float(config["w3_rad_s"]) / (2.0 * np.pi)
    burn_frac = float(
        burn_frac_override if burn_frac_override is not None else config.get("burn_frac", 0.25)
    )
    T = float(t_final_override if t_final_override is not None else config.get("t_final", 8.0))
    nbw_rel = float(nbw_rel_override if nbw_rel_override is not None else config["nbw_rel"])
    welch_nperseg = int(
        welch_nperseg_override if welch_nperseg_override is not None else config["welch_nperseg"]
    )
    welch_noverlap = int(
        welch_noverlap_override if welch_noverlap_override is not None else config["welch_noverlap"]
    )
    psd_norm = psd_norm.lower()
    return_onesided = psd_norm == "onesided"

    if theta_values is not None:
        thetas = sorted(theta_values)
        peak_checker = lambda val: abs(val - 1.0) < 1e-9
        seeds_peak_cfg = seeds_peak
        seeds_side_cfg = seeds_flank
    else:
        thetas_peak = set(float(theta) for theta in config.get("thetas_peak", []))
        thetas_side = set(float(theta) for theta in config.get("thetas_side", []))
        seeds_peak_cfg = int(config.get("seeds_peak", seeds_peak))
        seeds_side_cfg = int(config.get("seeds_side", seeds_flank))
        thetas = sorted(thetas_peak.union(thetas_side))
        peak_checker = lambda val: val in thetas_peak
    if not thetas:
        raise ValueError("No Θ values defined in configuration.")

    results_rows: List[Dict[str, float]] = []
    rng = np.random.default_rng(base_seed)
    baseband_data = {
        "ou": defaultdict(list),
        "psd_matched": defaultdict(list),
    }
    narrowband_data = {
        "ou": defaultdict(list),
        "psd_matched": defaultdict(list),
    }
    psd_store: Dict[float, Dict[str, List[np.ndarray] | np.ndarray]] = {}

    for theta in thetas:
        tau_B = theta / W1
        n_trials = seeds_peak_cfg if peak_checker(theta) else seeds_side_cfg

        for trial in range(n_trials):
            seed_val = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))

            xi = ou_process_exact(T, dt, tau_B, target_variance=noise_param, seed=seed_val)
            start_ou = time.perf_counter()
            times, _, _, x3_ou = run_hierarchy(
                tau_B=tau_B,
                T=T,
                dt=dt,
                calibration_mode=calibration_mode,
                noise_param=noise_param,
                burn_in_fraction=0.0,
                noise_override=xi,
            )
            wall_s_ou = time.perf_counter() - start_ou

            burn_idx = int(burn_frac * x3_ou.size)
            x3_ou = x3_ou[burn_idx:]
            ode_steps_ou = x3_ou.size
            fs_meas = 1.0 / dt
            nperseg_eff, noverlap_eff = _resolve_welch_params(xi.size, welch_nperseg, welch_noverlap)

            bb_ou = baseband_power(
                x3_ou,
                fs_hz=fs_meas,
                w3_rad_s=float(config["w3_rad_s"]),
                lp_frac=float(config["demod_lp_frac_of_w3"]),
            )
            nb_ou = narrowband_psd_power(
                x3_ou,
                fs_hz=fs_meas,
                w3_rad_s=float(config["w3_rad_s"]),
                rel_bandwidth=nbw_rel,
                nperseg=nperseg_eff,
                noverlap=noverlap_eff,
            )
            baseband_data["ou"][theta].append(bb_ou)
            narrowband_data["ou"][theta].append(nb_ou)

            surrogate = psd_matched_surrogate(xi, rng=rng)
            surrogate -= surrogate.mean()
            std_sur = np.std(surrogate)
            std_xi = np.std(xi)
            if std_sur > 0:
                surrogate *= std_xi / std_sur

            start_surr = time.perf_counter()
            _, _, _, x3_surr = run_hierarchy(
                tau_B=tau_B,
                T=T,
                dt=dt,
                calibration_mode=calibration_mode,
                noise_param=noise_param,
                burn_in_fraction=0.0,
                noise_override=surrogate,
            )
            wall_s_surr = time.perf_counter() - start_surr
            x3_surr = x3_surr[burn_idx:]
            ode_steps_surr = x3_surr.size
            bb_surr = baseband_power(
                x3_surr,
                fs_hz=fs_meas,
                w3_rad_s=float(config["w3_rad_s"]),
                lp_frac=float(config["demod_lp_frac_of_w3"]),
            )
            nb_surr = narrowband_psd_power(
                x3_surr,
                fs_hz=fs_meas,
                w3_rad_s=float(config["w3_rad_s"]),
                rel_bandwidth=nbw_rel,
                nperseg=nperseg_eff,
                noverlap=noverlap_eff,
            )
            baseband_data["psd_matched"][theta].append(bb_surr)
            narrowband_data["psd_matched"][theta].append(nb_surr)

            freq_ou, psd_ou = signal.welch(
                xi,
                fs=fs_hz,
                window="hann",
                nperseg=nperseg_eff,
                noverlap=noverlap_eff,
                detrend=False,
                scaling="density",
                return_onesided=return_onesided,
            )
            freq_sur, psd_sur = signal.welch(
                surrogate,
                fs=fs_hz,
                window="hann",
                nperseg=nperseg_eff,
                noverlap=noverlap_eff,
                detrend=False,
                scaling="density",
                return_onesided=return_onesided,
            )
            store = psd_store.setdefault(theta, {"freq": None, "ou": [], "psd": []})
            if store["freq"] is None:
                store["freq"] = freq_ou
            elif not np.allclose(store["freq"], freq_ou):
                raise RuntimeError("Frequency grid mismatch between PSD estimates.")
            if not np.allclose(freq_ou, freq_sur):
                raise RuntimeError("PSD frequency grids do not align.")
            store["ou"].append(psd_ou)
            store["psd"].append(psd_sur)

        metadata_common = {
            "theta": float(theta),
            "seed": seed_val,
            "fs_hz": fs_hz,
            "w3_rad_s": float(config["w3_rad_s"]),
            "demod_lp_frac": float(config["demod_lp_frac_of_w3"]),
            "nbw_rel": nbw_rel,
            "welch_nperseg": nperseg_eff,
            "welch_noverlap": noverlap_eff,
            "config_hash": config_hash,
            "psd_norm": psd_norm,
            "run_id": run_id,
            "holm_family": "classical_ou_vs_surrogate",
        }

        results_rows.append({
            **metadata_common,
            "model": "classical",
            "condition": "ou",
            "metric": "baseband_power",
            "value": bb_ou,
            "is_surrogate": 0,
            "psd_nrmse": np.nan,
            "wall_s": wall_s_ou,
            "ode_steps": float(ode_steps_ou),
            "timestamp": _utc_now(),
            "status": "ok",
            "failure_reason": "",
        })
        results_rows.append({
            **metadata_common,
            "model": "classical",
            "condition": "psd_matched",
            "metric": "baseband_power",
            "value": bb_surr,
            "is_surrogate": 1,
            "psd_nrmse": np.nan,
            "wall_s": wall_s_surr,
            "ode_steps": float(ode_steps_surr),
            "timestamp": _utc_now(),
            "status": "ok",
            "failure_reason": "",
        })
        results_rows.append({
            **metadata_common,
            "model": "classical",
            "condition": "ou",
            "metric": "narrowband_power",
            "value": nb_ou,
            "is_surrogate": 0,
            "psd_nrmse": np.nan,
            "wall_s": wall_s_ou,
            "ode_steps": float(ode_steps_ou),
            "timestamp": _utc_now(),
            "status": "ok",
            "failure_reason": "",
        })
        results_rows.append({
            **metadata_common,
            "model": "classical",
            "condition": "psd_matched",
            "metric": "narrowband_power",
            "value": nb_surr,
            "is_surrogate": 1,
            "psd_nrmse": np.nan,
            "wall_s": wall_s_surr,
            "ode_steps": float(ode_steps_surr),
            "timestamp": _utc_now(),
            "status": "ok",
            "failure_reason": "",
        })

    summary_records: List[Dict[str, float]] = []
    psd_nrmse: Dict[float, float] = {}
    for theta in thetas:
        ou_vals = np.asarray(baseband_data["ou"].get(theta, []), dtype=float)
        surrogate_vals = np.asarray(baseband_data["psd_matched"].get(theta, []), dtype=float)
        if ou_vals.size == 0 or surrogate_vals.size == 0:
            continue
        if ou_vals.size != surrogate_vals.size:
            print(f"Warning: Sample size mismatch at theta={theta} (OU={ou_vals.size}, surrogate={surrogate_vals.size})")
            continue

        # Paired statistics: each surrogate is built from its corresponding OU trace
        delta = ou_vals - surrogate_vals
        ci_ou = bootstrap_ci(ou_vals)
        ci_surr = bootstrap_ci(surrogate_vals)
        ci_delta = bootstrap_ci(delta)  # 95% CI on paired difference

        # Use paired Cohen's d_z instead of unpaired Cohen's d
        effect = paired_cohen_d(ou_vals, surrogate_vals)

        # Use paired t-test instead of independent samples t-test
        t_stat, p_value = stats.ttest_rel(ou_vals, surrogate_vals)

        # TOST equivalence test (ε = 2% of OU mean)
        tost_pass, tost_p_lower, tost_p_upper = tost_equivalence_paired(ou_vals, surrogate_vals, epsilon_rel=0.02)

        nrmse = np.nan
        if theta in psd_store:
            freq = np.asarray(psd_store[theta]["freq"], dtype=float)
            ou_stack = np.vstack(psd_store[theta]["ou"])
            surr_stack = np.vstack(psd_store[theta]["psd"])
            ou_mean = np.mean(ou_stack, axis=0)
            surr_mean = np.mean(surr_stack, axis=0)
            band_low = max(0.0, w3_hz * (1.0 - nbw_rel))
            band_high = w3_hz * (1.0 + nbw_rel)
            freq_eval = np.abs(freq)
            mask = (freq_eval >= band_low) & (freq_eval <= band_high) & (freq >= 0)
            if not np.any(mask):
                mask = freq >= 0
            diff = surr_mean[mask] - ou_mean[mask]
            denom = np.sqrt(np.mean(ou_mean[mask] ** 2)) if np.any(mask) else 1.0
            if denom == 0:
                denom = 1.0
            nrmse = float(np.sqrt(np.mean(diff**2)) / denom)
            psd_nrmse[theta] = nrmse

        summary_records.append(
            {
                "theta": theta,
                "n_samples": float(ou_vals.size),
                "mean_ou": float(np.mean(ou_vals)),
                "ci_ou_low": ci_ou[0],
                "ci_ou_high": ci_ou[1],
                "mean_psd_matched": float(np.mean(surrogate_vals)),
                "ci_psd_low": ci_surr[0],
                "ci_psd_high": ci_surr[1],
                "mean_delta": float(np.mean(delta)),
                "ci_delta_low": ci_delta[0],
                "ci_delta_high": ci_delta[1],
                "cohen_d": effect,  # Paired d_z (descriptive)
                "p_value": float(p_value),  # Paired t-test
                "tost_pass": int(tost_pass),
                "tost_p_lower": float(tost_p_lower),
                "tost_p_upper": float(tost_p_upper),
                "psd_nrmse": nrmse,
                "timestamp": _utc_now(),
                "status": "ok",
                "holm_family": "classical_ou_vs_surrogate",
            }
        )

    if summary_records:
        sorted_indices = sorted(range(len(summary_records)), key=lambda i: summary_records[i]["p_value"])
        m = len(summary_records)
        prev = 1.0
        for rank, idx in enumerate(sorted_indices, start=1):
            p = summary_records[idx]["p_value"]
            adj = min(1.0, (m - rank + 1) * p)
            adj = min(adj, prev)
            summary_records[idx]["p_value_holm"] = adj
            prev = adj
        # fill any missing keys
        for record in summary_records:
            record.setdefault("p_value_holm", min(1.0, record["p_value"]))

    summary_path = output_dir / "classical_psd_control_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "theta",
            "n_samples",
            "mean_ou",
            "ci_ou_low",
            "ci_ou_high",
            "mean_psd_matched",
            "ci_psd_low",
            "ci_psd_high",
            "mean_delta",
            "ci_delta_low",
            "ci_delta_high",
            "cohen_d",
            "p_value",
            "p_value_holm",
            "tost_pass",
            "tost_p_lower",
            "tost_p_upper",
            "psd_nrmse",
            "holm_family",
            "timestamp",
            "status",
        ])
        for record in summary_records:
            writer.writerow([
                record["theta"],
                record["n_samples"],
                record["mean_ou"],
                record["ci_ou_low"],
                record["ci_ou_high"],
                record["mean_psd_matched"],
                record["ci_psd_low"],
                record["ci_psd_high"],
                record.get("mean_delta", np.nan),
                record.get("ci_delta_low", np.nan),
                record.get("ci_delta_high", np.nan),
                record["cohen_d"],
                record["p_value"],
                record["p_value_holm"],
                record.get("tost_pass", 0),
                record.get("tost_p_lower", np.nan),
                record.get("tost_p_upper", np.nan),
                record["psd_nrmse"],
                record.get("holm_family", ""),
                record.get("timestamp", ""),
                record.get("status", ""),
            ])

    fig, ax = plt.subplots(figsize=(6, 4))
    for condition in ("ou", "psd_matched"):
        theta_arr = np.array(sorted(baseband_data[condition].keys()), dtype=float)
        means = [np.mean(baseband_data[condition][theta]) for theta in theta_arr]
        mean_arr = np.array(means)
        cis = [bootstrap_ci(np.asarray(baseband_data[condition][theta], dtype=float)) for theta in theta_arr]
        ci_low_arr = np.array([ci[0] for ci in cis])
        ci_high_arr = np.array([ci[1] for ci in cis])
        ax.plot(theta_arr, mean_arr, marker="o", label=condition.replace("_", " "))
        ax.fill_between(theta_arr, ci_low_arr, ci_high_arr, alpha=0.2)

    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"$\Theta = \omega_1 \tau_B$")
    ax.set_ylabel("Baseband power")
    ax.set_title("Classical OU vs PSD-matched surrogate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig_path = output_dir / "figure_classical_psd_control.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    for theta, data in psd_store.items():
        freq = np.asarray(data["freq"], dtype=float)
        ou_stack = np.vstack(data["ou"]) if data["ou"] else np.empty((0, freq.shape[0]))
        surr_stack = np.vstack(data["psd"]) if data["psd"] else np.empty((0, freq.shape[0]))
        ou_mean = np.mean(ou_stack, axis=0)
        surr_mean = np.mean(surr_stack, axis=0)

        fig_psd, ax_psd = plt.subplots(figsize=(6, 4))
        ax_psd.loglog(freq, ou_mean, label="OU drive", color="tab:blue")
        ax_psd.loglog(freq, surr_mean, label="PSD-matched surrogate", color="tab:orange", linestyle="--")
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("PSD")
        ax_psd.set_title(f"PSD overlay (Θ = {theta:.2f})")
        ax_psd.grid(True, which="both", alpha=0.3)
        ax_psd.legend()
        overlay_path = output_dir / f"psd_overlay_theta_{theta:.3f}.png"
        fig_psd.tight_layout()
        fig_psd.savefig(overlay_path, dpi=300)
        plt.close(fig_psd)

        residual = surr_mean - ou_mean
        residual_std = (
            np.std(surr_stack - ou_stack, axis=0)
            if ou_stack.size and surr_stack.size
            else np.zeros_like(residual)
        )
        fig_res, ax_res = plt.subplots(figsize=(6, 4))
        ax_res.semilogx(freq, residual, color="tab:purple", label="Surrogate - OU")
        ax_res.fill_between(
            freq,
            residual - residual_std,
            residual + residual_std,
            color="tab:purple",
            alpha=0.2,
            label="±1σ",
        )
        ax_res.set_xlabel("Frequency (Hz)")
        ax_res.set_ylabel("Residual PSD")
        ax_res.set_title(f"PSD residual (Θ = {theta:.2f})")
        ax_res.grid(True, which="both", alpha=0.3)
        ax_res.legend()
        residual_path = output_dir / f"psd_residual_theta_{theta:.3f}.png"
        fig_res.tight_layout()
        fig_res.savefig(residual_path, dpi=300)
        plt.close(fig_res)

        print(f"✓ PSD overlay saved to {overlay_path}")
        print(f"✓ PSD residual saved to {residual_path}")

        results_rows.append({
            "model": "classical",
            "condition": "psd_overlay",
            "metric": "psd_nrmse",
            "theta": float(theta),
            "seed": -1,
            "value": psd_nrmse.get(theta, np.nan),
            "fs_hz": fs_hz,
            "w3_rad_s": float(config["w3_rad_s"]),
            "demod_lp_frac": float(config["demod_lp_frac_of_w3"]),
            "nbw_rel": nbw_rel,
            "welch_nperseg": welch_nperseg,
            "welch_noverlap": welch_noverlap,
            "psd_norm": psd_norm,
            "is_surrogate": 2,
            "psd_nrmse": psd_nrmse.get(theta, np.nan),
            "config_hash": config_hash,
            "wall_s": np.nan,
            "ode_steps": np.nan,
            "timestamp": _utc_now(),
            "run_id": run_id,
            "holm_family": "classical_ou_vs_surrogate",
            "status": "ok",
            "failure_reason": "",
        })

    csv_path = output_dir / "classical_psd_control_results.csv"
    fieldnames = [
        "model",
        "condition",
        "theta",
        "seed",
        "metric",
        "value",
        "fs_hz",
        "w3_rad_s",
        "demod_lp_frac",
        "nbw_rel",
        "welch_nperseg",
        "welch_noverlap",
        "psd_norm",
        "is_surrogate",
        "psd_nrmse",
        "config_hash",
        "run_id",
        "holm_family",
        "wall_s",
        "ode_steps",
        "timestamp",
        "status",
        "failure_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"✓ Classical PSD control results saved to {csv_path}")
    print(f"✓ Summary saved to {summary_path}")
    print(f"✓ Figure saved to {fig_path}")

    manifest = {
        "timestamp": _utc_now(),
        "config_hash": config_hash,
        "calibration": calibration_mode,
        "psd_norm": psd_norm,
        "noise_param": noise_param,
        "thetas": list(thetas),
        "seeds": {
            "peak": seeds_peak_cfg,
            "flank": seeds_side_cfg,
        },
        "fs_hz": fs_hz,
        "t_final": T,
        "burn_frac": burn_frac,
        "nbw_rel": nbw_rel,
        "welch": {
            "nperseg": welch_nperseg,
            "noverlap": welch_noverlap,
        },
        "run_id": run_id,
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"✓ MANIFEST saved to {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical PSD-matched control experiment")
    parser.add_argument("--output_dir", type=Path, default=Path("results/classical_psd_control"))
    parser.add_argument("--calibration", type=str, default="equal_variance")
    parser.add_argument("--noise_param", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--thetas", type=str, help="Comma-separated Θ values to evaluate.")
    parser.add_argument("--seeds_peak", type=int, default=20)
    parser.add_argument("--seeds_flank", type=int, default=8)
    parser.add_argument("--welch_nperseg", type=int)
    parser.add_argument("--welch_noverlap", type=int)
    parser.add_argument("--nbw_rel", type=float)
    parser.add_argument("--burn_frac", type=float)
    parser.add_argument("--t_final", type=float)
    parser.add_argument(
        "--psd_norm",
        type=str,
        choices=["onesided", "twosided"],
        default="onesided",
        help="PSD normalization convention (onesided or twosided).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_hash = load_config_with_hash()
    theta_values = None
    if args.thetas:
        theta_values = [float(x.strip()) for x in args.thetas.split(",") if x.strip()]
    run_classical_chain(
        config=config,
        config_hash=config_hash,
        output_dir=args.output_dir,
        calibration_mode=args.calibration,
        noise_param=args.noise_param,
        base_seed=args.seed,
        theta_values=theta_values,
        seeds_peak=args.seeds_peak,
        seeds_flank=args.seeds_flank,
        t_final_override=args.t_final,
        burn_frac_override=args.burn_frac,
        welch_nperseg_override=args.welch_nperseg,
        welch_noverlap_override=args.welch_noverlap,
        nbw_rel_override=args.nbw_rel,
        psd_norm=args.psd_norm,
    )


if __name__ == "__main__":
    main()
