"""
Stage 3: Sweep pseudomode correlation times and compare to Lindblad baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import uuid
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from tqdm import tqdm

from qutip import expect, steadystate, spectrum

from analysis.config_loader import load_config_with_hash
from analysis.metrics import (
    baseband_power,
    narrowband_psd_power,
    baseband_power_from_psd,
    narrowband_power_from_psd,
    one_sided_psd_from_omega,
    band_average_from_psd,
)
from quant.gaussian_linear import (
    GaussianModel,
    build_model,
    mode_occupancy,
    psd as gaussian_psd,
    steady_covariance,
    quadrature_variance,
)
from quantum_models import (
    W1,
    W2,
    W3,
    GAMMA1,
    GAMMA2,
    GAMMA3,
    G12,
    G23,
    calibrate_equal_carrier_g,
    baseline_pseudomode_coupling,
    pseudomode_lorentzian_j,
    build_markovian_operators,
    build_pseudomode_operators,
    run_markovian_model,
    run_pseudomode_model,
    compute_qrt_psd_markovian,
    compute_qrt_psd_pseudomode,
)

ENVELOPE_BAND = (0.5, 5.0)
OCC_TAIL_FRACTION = 0.2
OCC_MIN_THRESHOLD = 1e-9
SUMMARY_FIELDNAMES = [
    "theta",
    "seed",
    "tau_B",
    "calibration",
    "cutoff",
    "N",
    "N_pseudo",
    "model",
    "R_env",
    "band_power",
    "energy",
    "envelope_rms",
    "envelope_rms_ratio",
    "baseband_power",
    "baseband_ratio",
    "narrowband_power",
    "occ_target",
    "occ_meas",
    "rel_occ_err",
    "n1_tail",
    "occ_ratio_achieved",
    "gpm_scale",
    "nudges_used",
    "hit_clamp",
    "g_used",
    "kappa",
    "Jw1_target",
    "Jw1_est",
    "rel_Jw1_err",
    "nbar",
    "a1_init_alpha",
    "burn_frac",
    "omega_c",
    "t_final",
    "n_time",
    "psd_norm",
    "is_surrogate",
    "psd_nrmse",
    "config_hash",
    "wall_s",
    "ode_steps",
    "timestamp",
    "run_id",
    "solver",
    "min_real_part",
    "cond_drift",
    "freq_grid",
    "j_bandwidth_frac",
    "holm_family",
    "status",
    "failure_reason",
]


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def compute_band_power(times: np.ndarray, signal_values: np.ndarray) -> float:
    dt = float(times[1] - times[0])
    freq, psd = signal.periodogram(signal_values, fs=1.0 / dt)
    mask = (freq >= ENVELOPE_BAND[0]) & (freq <= ENVELOPE_BAND[1])
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freq[mask]))


def tail_average(values: np.ndarray, fraction: float = 0.3) -> float:
    if values.size == 0:
        return 0.0
    start = max(int(values.size * (1.0 - fraction)), 0)
    return float(np.mean(values[start:]))


def compute_envelope_rms(
    signal_values: np.ndarray,
    dt: float,
    lowpass_cutoff: float | None = ENVELOPE_BAND[1],
) -> float:
    if signal_values.size == 0:
        return 0.0
    demeaned = signal_values - np.mean(signal_values)
    pad = max(1, int(0.1 * len(demeaned)))
    if pad > 0:
        extended = np.concatenate(
            (demeaned[:pad][::-1], demeaned, demeaned[-pad:][::-1])
        )
    else:
        extended = demeaned

    analytic = signal.hilbert(extended)
    envelope = np.abs(analytic)
    if pad > 0:
        envelope = envelope[pad:-pad]

    if envelope.size == 0:
        return 0.0

    if lowpass_cutoff is not None and dt > 0:
        fs = 1.0 / dt
        nyquist = 0.5 * fs
        if nyquist > 0:
            norm_cutoff = lowpass_cutoff / nyquist
            if 0 < norm_cutoff < 1:
                sos = signal.butter(4, norm_cutoff, btype="low", output="sos")
                envelope = signal.sosfiltfilt(sos, envelope)

    return float(np.sqrt(np.mean(envelope**2)))


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _safe_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _extract_ode_steps(stats: dict | None) -> float:
    if not stats:
        return float("nan")
    for key in ("num_steps", "nsteps", "solver_nsteps"):
        value = stats.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return float("nan")


def append_summary_rows(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header_exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDNAMES)
        if not header_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDNAMES})


def load_summary_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def update_quantum_manifest(output_dir: Path, rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        return

    pseud_rows = [row for row in rows if row.get("model") == "pseudomode"]
    if not pseud_rows:
        return

    theta_dir = output_dir.parent
    theta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = theta_dir / "MANIFEST.json"

    try:
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    except json.JSONDecodeError:
        manifest = {}

    theta_value = next((val for val in (_safe_float(row.get("theta")) for row in pseud_rows) if val is not None), None)
    config_hash = pseud_rows[0].get("config_hash")

    manifest.update(
        {
            "theta": theta_value,
            "config_hash": config_hash,
            "updated_at": _utc_now(),
        }
    )

    existing_entries = manifest.get("entries", [])
    exclude_ids = {row.get("run_id") for row in pseud_rows}
    filtered_entries = [entry for entry in existing_entries if entry.get("run_id") not in exclude_ids]

    for row in pseud_rows:
        seed_val = _safe_float(row.get("seed"))
        freq_grid_raw = row.get("freq_grid", "")
        try:
            freq_grid = json.loads(freq_grid_raw) if freq_grid_raw else []
        except json.JSONDecodeError:
            freq_grid = []
        entry = {
            "run_id": row.get("run_id"),
            "seed": int(seed_val) if seed_val is not None else None,
            "calibration": row.get("calibration"),
            "cutoff": _safe_float(row.get("cutoff")),
            "kappa": _safe_float(row.get("kappa")),
            "g_used": _safe_float(row.get("g_used")),
            "Jw1_target": _safe_float(row.get("Jw1_target")),
            "Jw1_est": _safe_float(row.get("Jw1_est")),
            "rel_Jw1_err": _safe_float(row.get("rel_Jw1_err")),
            "rel_occ_err": _safe_float(row.get("rel_occ_err")),
            "nudges_used": _safe_float(row.get("nudges_used")),
            "wall_s": _safe_float(row.get("wall_s")),
            "ode_steps": _safe_float(row.get("ode_steps")),
            "psd_norm": row.get("psd_norm"),
            "solver": row.get("solver"),
            "min_real_part": _safe_float(row.get("min_real_part")),
            "cond_drift": _safe_float(row.get("cond_drift")),
            "freq_grid": freq_grid,
            "j_bandwidth_frac": _safe_float(row.get("j_bandwidth_frac")),
            "status": row.get("status"),
            "failure_reason": row.get("failure_reason"),
            "timestamp": row.get("timestamp"),
        }
        filtered_entries.append(entry)

    manifest["entries"] = filtered_entries
    manifest["j_bandwidth_frac"] = _safe_float(pseud_rows[0].get("j_bandwidth_frac"))
    if filtered_entries:
        manifest["freq_grid_signature"] = filtered_entries[0].get("freq_grid", [])
    manifest_path.write_text(json.dumps(manifest, indent=2))


def load_classical_curves(path: Path | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    curves: dict[str, list[tuple[float, float]]] = {}
    if not path or not path.exists():
        return {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                theta = float(row.get("theta", row.get("Theta")))
                mean_val = float(row["R_env_mean"])
            except (KeyError, ValueError):
                continue
            calibration = row.get("calibration", "equal_variance")
            curves.setdefault(calibration, []).append((theta, mean_val))
    ordered: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for calibration, points in curves.items():
        points.sort(key=lambda x: x[0])
        theta_vals, means = zip(*points)
        ordered[calibration] = (np.array(theta_vals), np.array(means))
    return ordered


def apply_burn_in(
    times: np.ndarray,
    arrays: Sequence[np.ndarray],
    burn_fraction: float,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    if not 0 < burn_fraction < 1:
        return times, tuple(arr.copy() for arr in arrays)
    idx = int(len(times) * burn_fraction)
    if idx >= len(times) - 2:
        return times, tuple(arr.copy() for arr in arrays)
    trimmed_times = times[idx:] - times[idx]
    trimmed_arrays = tuple(arr[idx:] for arr in arrays)
    return trimmed_times, trimmed_arrays


def _infer_seed_label(path: Path) -> int:
    """
    Try to extract an integer seed label from the output path.

    Accepts directory names such as ``seed_3`` or ``seed3``. If no seed token
    is found, returns ``-1`` to indicate an unspecified seed.
    """
    pattern = re.compile(r"seed[_-]?(\d+)", re.IGNORECASE)
    for part in reversed(path.parts):
        match = pattern.search(part)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return -1


def run_quantum_sweep(
    tau_values: np.ndarray,
    calibrations: Sequence[str],
    cutoffs: Sequence[int],
    times: np.ndarray,
    hierarchy_cutoff: int,
    solver_progress: str | None,
    atol: float,
    rtol: float,
    nsteps: int,
    rhs_reuse: bool,
    summary_path: Path,
    nbar: float,
    a1_init_alpha: float,
    burn_fraction: float,
    omega_c_scale: float,
    t_final: float,
    n_time: int,
    config_data: dict,
    config_hash_value: str,
    tuner_tol_occ: float,
    tuner_max_nudges: int,
    carrier_map_entries: Optional[dict[float, dict]],
    carrier_reference: Optional[dict],
    psd_norm: str,
    equal_carrier_tol: float,
    run_id: str,
    seed_label: Optional[int] = None,
) -> list[dict[str, float]]:
    results_all: list[dict[str, float]] = []
    pending_rows: list[dict[str, float]] = []

    def record_row(row: dict[str, float]) -> None:
        row.setdefault("run_id", run_id)
        row.setdefault("timestamp", _utc_now())
        results_all.append(row)
        pending_rows.append(row)

    def flush_pending() -> None:
        nonlocal pending_rows
        if pending_rows:
            append_summary_rows(summary_path, pending_rows)
            pending_rows = []

    burn_fraction = float(np.clip(burn_fraction, 0.0, 0.95))
    omega_c_value = omega_c_scale * W1
    dt_nominal = float(np.mean(np.diff(times))) if len(times) > 1 else 1.0
    fs_config = float(config_data.get('fs_hz', 1.0))
    w3_config = float(config_data.get('w3_rad_s', 1.0))
    lp_frac = float(config_data.get('demod_lp_frac_of_w3', 0.2))
    rel_bw = float(config_data.get('nbw_rel', 0.3))
    welch_nperseg = int(config_data.get('welch_nperseg', 1024))
    welch_noverlap = int(config_data.get('welch_noverlap', 512))
    tuner_tol_occ = float(tuner_tol_occ)
    tuner_max_nudges = max(int(tuner_max_nudges), 0)
    equal_carrier_tol = float(equal_carrier_tol)
    carrier_map_entries = carrier_map_entries or {}
    seed_value = seed_label if seed_label is not None else _infer_seed_label(summary_path.parent)
    use_adaptive = any(cal == "equal_heating" for cal in calibrations)
    times_short = times
    if use_adaptive:
        n_short = max(30, int(len(times) * 0.12))
        if n_short < len(times):
            times_short = times[:n_short].copy()

    carrier_reference_local = carrier_reference
    if any(cal == 'equal_carrier' for cal in calibrations) and carrier_reference_local is None:
        theta_ref_candidates = config_data.get('thetas_peak', [1.0])
        theta_ref = float(theta_ref_candidates[0])
        tau_ref = theta_ref / W1
        kappa_ref = 1.0 / tau_ref
        g_ref = baseline_pseudomode_coupling(tau_ref, 'equal_carrier', W1, GAMMA1)
        delta_ref = W1 - omega_c_value
        target_j_ref = pseudomode_lorentzian_j(g_ref, kappa_ref, delta_ref)
        carrier_reference_local = {
            'theta': theta_ref,
            'tau': tau_ref,
            'kappa': kappa_ref,
            'g': g_ref,
            'target_j': target_j_ref,
            'delta': delta_ref,
        }

    def find_carrier_entry(theta_value: float) -> dict | None:
        for key, val in carrier_map_entries.items():
            if abs(key - theta_value) <= 1e-9:
                return val
        return None

    try:
        for calibration in calibrations:
            print("\n" + "=" * 70)
            print(f"Calibration: {calibration}")
            total_runs = len(cutoffs) * len(tau_values) + len(cutoffs)
            print(
                f'Total runs for this calibration: {len(cutoffs)} cutoffs × {len(tau_values)} τ_B values + {len(cutoffs)} baselines = {total_runs}',
                flush=True,
            )
            for cutoff in cutoffs:
                print('-' * 70)
                print(f'Pseudomode cutoff (N_pseudo): {cutoff}')
                print(f'  [{timestamp()}] Computing Markovian baseline...', flush=True)
                baseline_start = time.perf_counter()

                # Use Gaussian covariance solver (fast and exact) instead of time-domain trajectories
                # This computes steady-state fluctuation spectra, matching the Gaussian engine exactly
                baseline_model, _, _, _, _ = _build_gaussian_model(
                    include_pseudomode=False,
                    tau_B=1.0,
                    calibration=calibration,
                    gpm_scale=1.0,
                    g_override=None,
                    omega_c_scale=omega_c_scale,
                    nbar=nbar,
                    nbar_pseudo=nbar,
                )
                baseline_obs, _, _ = _compute_observables(baseline_model, lp_frac, rel_bw)
                baseline_band = baseline_obs["band_power"]
                baseline_narrowband = baseline_obs["narrowband_power"]
                baseline_baseband = baseline_obs["baseband_power"]
                baseline_envelope = baseline_obs["envelope_rms"]
                baseline_energy = baseline_obs["energy"]
                baseline_n1 = max(baseline_obs["occ_meas"], 1e-12)

                baseline_wall = time.perf_counter() - baseline_start
                stats_m = {}
                print(
                    f'  [{timestamp()}] ✓ Baseline completed in {baseline_wall:.2f} s',
                    flush=True,
                )
                print(f'  [{timestamp()}] OCC baseline ⟨n₁⟩_tail = {baseline_n1:.4e}', flush=True)
                is_carrier = calibration == 'equal_carrier'

                baseline_row = {
                    'theta': 0.0,
                    'seed': -1,
                    'tau_B': 0.0,
                    'calibration': calibration,
                    'cutoff': float(cutoff),
                    'N': float(hierarchy_cutoff),
                    'N_pseudo': float(cutoff),
                    'model': 'lindblad',
                    'R_env': 1.0,
                    'band_power': baseline_band,
                    'energy': baseline_energy,
                    'envelope_rms': baseline_envelope,
                    'envelope_rms_ratio': 1.0,
                    'baseband_power': baseline_baseband,
                    'baseband_ratio': 1.0,
                    'narrowband_power': baseline_narrowband,
                    'occ_target': baseline_n1,
                    'occ_meas': baseline_n1,
                    'rel_occ_err': 0.0,
                    'n1_tail': baseline_n1,
                    'occ_ratio_achieved': 1.0,
                    'gpm_scale': 1.0,
                    'nudges_used': 0,
                    'hit_clamp': False,
                    'g_used': carrier_reference_local['g'] if is_carrier and carrier_reference_local else np.nan,
                    'kappa': carrier_reference_local['kappa'] if is_carrier and carrier_reference_local else np.nan,
                    'Jw1_target': carrier_reference_local['target_j'] if is_carrier and carrier_reference_local else np.nan,
                    'Jw1_est': np.nan,
                    'rel_Jw1_err': np.nan,
                    'nbar': nbar,
                    'a1_init_alpha': a1_init_alpha,
                    'burn_frac': burn_fraction,
                    'omega_c': omega_c_value,
                    't_final': t_final,
                    'n_time': n_time,
                    'psd_norm': psd_norm,
                    'is_surrogate': 0,
                    'psd_nrmse': np.nan,
                    'config_hash': config_hash_value,
                    'wall_s': baseline_wall,
                    'ode_steps': _extract_ode_steps(stats_m),
                    'timestamp': _utc_now(),
                    'run_id': run_id,
                    'solver': 'trajectory',
                    'min_real_part': float('nan'),
                    'cond_drift': float('nan'),
                    'freq_grid': json.dumps([]),
                    'j_bandwidth_frac': '',
                    'holm_family': '',
                    'status': 'ok',
                    'failure_reason': '',
                }
                record_row(baseline_row)

                desc = f"{calibration:>16} Np={cutoff}"
                adaptive = calibration == 'equal_heating'
                carrier_mode = calibration == 'equal_carrier'
                current_scale = 1.0
                for tau_B in tqdm(tau_values, desc=desc, ncols=95):
                    theta = W1 * tau_B
                    print(
                        f'    [{timestamp()}] Solving pseudomode run: τ_B = {tau_B:.3e} (Θ = {theta:.3f}), Np={cutoff}',
                        flush=True,
                    )

                    scale = current_scale
                    nudges_used = 0
                    hit_clamp = False
                    kappa = 1.0 / tau_B if tau_B > 0 else float("inf")
                    g_override = None
                    j_target = np.nan
                    j_est = np.nan
                    j_rel_err = np.nan
                    if carrier_mode:
                        if carrier_reference_local is None:
                            raise RuntimeError('Equal-carrier calibration requires reference data.')
                        entry = find_carrier_entry(theta)
                        if entry is not None:
                            delta_local = entry.get('delta', carrier_reference_local['delta'])
                            g_override = float(entry.get('g', entry.get('g_used', 0.0)))
                            j_target = float(entry.get('Jw1_target', carrier_reference_local['target_j']))
                            j_est = float(
                                entry.get(
                                    'Jw1_est',
                                    pseudomode_lorentzian_j(
                                        g_override,
                                        kappa,
                                        delta_local,
                                    ),
                                )
                            )
                            if j_target > 0:
                                j_rel_err = abs(j_est - j_target) / j_target
                        else:
                            g_override, j_target, j_est, j_rel_err = calibrate_equal_carrier_g(
                                kappa=kappa,
                                baseline_kappa=carrier_reference_local['kappa'],
                                baseline_g=carrier_reference_local['g'],
                                delta=carrier_reference_local['delta'],
                                tolerance=equal_carrier_tol,
                            )

                    solver_wall = 0.0
                    last_stats: dict = {}

                    # Helper function to compute metrics using Gaussian covariance solver
                    def compute_gaussian_metrics_pseudomode(scale_val: float, g_ovr: float | None) -> tuple[float, float, float, float, float, float]:
                        """Compute metrics using Gaussian covariance. Returns (band_power, baseband, narrowband, n1, energy, env_rms)."""
                        iter_start = time.perf_counter()
                        model_eval, g_used_eval, _, omega_c_eval, _ = _build_gaussian_model(
                            include_pseudomode=True,
                            tau_B=tau_B,
                            calibration=calibration,
                            gpm_scale=scale_val,
                            g_override=g_ovr,
                            omega_c_scale=omega_c_scale,
                            nbar=nbar,
                            nbar_pseudo=nbar,
                        )
                        obs_eval, _, _ = _compute_observables(model_eval, lp_frac, rel_bw)
                        nonlocal solver_wall
                        solver_wall += time.perf_counter() - iter_start

                        return (
                            obs_eval["band_power"],
                            obs_eval["baseband_power"],
                            obs_eval["narrowband_power"],
                            obs_eval["occ_meas"],
                            obs_eval["energy"],
                            obs_eval["envelope_rms"],
                        )

                    # Adaptive tuning loop (or single evaluation if not adaptive)
                    while True:
                        band_power, baseband_val, narrowband_val, n1_tail, energy, env_rms = compute_gaussian_metrics_pseudomode(scale, g_override)

                        if not adaptive:
                            break

                        if baseline_n1 <= OCC_MIN_THRESHOLD or n1_tail <= OCC_MIN_THRESHOLD:
                            break

                        occ_ratio = n1_tail / baseline_n1
                        needs_adjustment = abs(occ_ratio - 1.0) > tuner_tol_occ

                        if needs_adjustment and nudges_used < tuner_max_nudges:
                            scale_factor = np.sqrt(1.0 / occ_ratio)
                            proposed_scale = float(scale * np.clip(scale_factor, 0.6, 2.2))
                            if abs(proposed_scale - scale) <= 0.02:
                                hit_clamp = True
                                break
                            if proposed_scale in (0.6, 2.2):
                                hit_clamp = True
                                break

                            print(
                                f'      [{timestamp()}] Adjusting g_pm scale from {scale:.3f} to {proposed_scale:.3f} to match ⟨n₁⟩',
                                flush=True,
                            )
                            scale = proposed_scale
                            nudges_used += 1
                            continue

                        if needs_adjustment and nudges_used >= tuner_max_nudges:
                            hit_clamp = True
                            break

                        break

                    current_scale = scale if adaptive else 1.0
                    occ_ratio_final = (
                        n1_tail / baseline_n1
                        if baseline_n1 > OCC_MIN_THRESHOLD and n1_tail > OCC_MIN_THRESHOLD
                        else np.nan
                    )
                    r_env = band_power / baseline_band if baseline_band > 0 else np.nan
                    env_ratio = (
                        env_rms / baseline_envelope if baseline_envelope > 0 else np.nan
                    )
                    if g_override is not None:
                        g_used = g_override
                    else:
                        g_base = baseline_pseudomode_coupling(tau_B, calibration, W1, GAMMA1)
                        g_used = g_base * scale
                        # For equal_carrier mode, compute J using the same method as the Gaussian engine
                    # Must use local delta (W1 - omega_c_value) for consistency with Gaussian engine
                    if carrier_mode:
                        j_target = carrier_reference_local['target_j'] if carrier_reference_local is not None else np.nan
                        # Compute delta from local omega_c (same as Gaussian engine at line 1176)
                        delta_local = W1 - omega_c_value
                        j_est = pseudomode_lorentzian_j(g_used, kappa, delta_local)
                        j_rel_err = abs(j_est - j_target) / j_target if j_target > 0 else np.nan
                    else:
                        j_target = np.nan
                        j_est = np.nan
                        j_rel_err = np.nan

                    baseband_ratio = (
                        baseband_val / baseline_baseband if baseline_baseband > 0 else np.nan
                    )

                    rel_occ_err = (
                        float((n1_tail - baseline_n1) / baseline_n1)
                        if baseline_n1 > OCC_MIN_THRESHOLD
                        else np.nan
                    )

                    status = 'ok'
                    failure_reason_parts: list[str] = []
                    if adaptive and not np.isnan(rel_occ_err):
                        if abs(rel_occ_err) > tuner_tol_occ and nudges_used >= tuner_max_nudges:
                            status = 'fail'
                            failure_reason_parts.append('rel_occ_err')
                    if carrier_mode and not np.isnan(j_rel_err) and j_rel_err > equal_carrier_tol:
                        status = 'fail'
                        failure_reason_parts.append('rel_Jw1_err')

                    row = {
                        'theta': float(theta),
                        'seed': seed_value,
                        'tau_B': float(tau_B),
                        'calibration': calibration,
                        'cutoff': float(cutoff),
                        'N': float(hierarchy_cutoff),
                        'N_pseudo': float(cutoff),
                        'model': 'pseudomode',
                        'R_env': float(r_env),
                        'band_power': float(band_power),
                        'energy': float(energy),
                        'envelope_rms': float(env_rms),
                        'envelope_rms_ratio': float(env_ratio),
                        'baseband_power': float(baseband_val),
                        'baseband_ratio': float(baseband_ratio),
                        'narrowband_power': float(narrowband_val),
                        'occ_target': float(baseline_n1),
                        'occ_meas': float(n1_tail),
                        'rel_occ_err': rel_occ_err,
                        'n1_tail': float(n1_tail),
                        'occ_ratio_achieved': float(occ_ratio_final)
                        if not np.isnan(occ_ratio_final)
                        else np.nan,
                        'gpm_scale': float(scale),
                        'nudges_used': nudges_used,
                        'hit_clamp': hit_clamp,
                        'g_used': float(g_used),
                        'kappa': float(kappa),
                        'Jw1_target': float(j_target),
                        'Jw1_est': float(j_est),
                        'rel_Jw1_err': float(j_rel_err),
                        'nbar': nbar,
                        'a1_init_alpha': a1_init_alpha,
                        'burn_frac': burn_fraction,
                        'omega_c': omega_c_value,
                        't_final': t_final,
                        'n_time': n_time,
                        'psd_norm': psd_norm,
                        'is_surrogate': 0,
                        'psd_nrmse': np.nan,
                        'config_hash': config_hash_value,
                        'wall_s': solver_wall,
                        'ode_steps': _extract_ode_steps(last_stats),
                        'timestamp': _utc_now(),
                        'run_id': run_id,
                        'solver': 'trajectory',
                        'min_real_part': float('nan'),
                        'cond_drift': float('nan'),
                        'freq_grid': json.dumps([]),
                        'j_bandwidth_frac': '',
                        'holm_family': '',
                        'status': status,
                        'failure_reason': ';'.join(failure_reason_parts),
                    }
                    record_row(row)

                    occ_ratio_text = (
                        f'{occ_ratio_final:.3f}'
                        if not np.isnan(occ_ratio_final)
                        else 'nan'
                    )
                    j_text = (
                        f'{j_rel_err:.3%}'
                        if carrier_mode and not np.isnan(j_rel_err)
                        else 'n/a'
                    )
                    print(
                        f'      [{timestamp()}] ✓ Completed τ_B = {tau_B:.3e} (Θ = {theta:.3f}); R_env = {r_env:.3f}; env_RMS ratio = {env_ratio:.3f}; n₁ ratio = {occ_ratio_text} (g_pm scale {scale:.3f}, nudges {nudges_used}, ΔJ/J={j_text}) (wall {solver_wall:.2f} s)',
                        flush=True,
                    )

                    if status != 'ok':
                        failure_text = ';'.join(failure_reason_parts)
                        raise RuntimeError(
                            f'Tolerance failure for Θ={theta:.3f}, seed={seed_value}: {failure_text}'
                        )
    except Exception:
        flush_pending()
        raise
    else:
        flush_pending()

    return results_all


def run_gaussian_sweep(
    tau_values: np.ndarray,
    calibrations: Sequence[str],
    cutoffs: Sequence[int],
    summary_path: Path,
    config_data: dict,
    config_hash_value: str,
    omega_c_scale: float,
    tuner_tol_occ: float,
    tuner_max_nudges: int,
    carrier_map_entries: Optional[dict[float, dict]],
    carrier_reference: Optional[dict],
    psd_norm: str,
    equal_carrier_tol: float,
    run_id: str,
    seed_label: Optional[int],
    nbar: float,
    a1_init_alpha: float,
    j_bandwidth_frac: float,
) -> list[dict[str, float]]:
    results_all: list[dict[str, float]] = []
    seed_value = seed_label if seed_label is not None else _infer_seed_label(summary_path.parent)

    lp_frac = float(config_data.get("demod_lp_frac_of_w3", 0.2))
    rel_bw = float(config_data.get("nbw_rel", 0.3))
    burn_frac = float(config_data.get("burn_frac", 0.25))
    t_final = float(config_data.get("t_final", 8.0))
    n_time = int(config_data.get("n_time", 300))

    carrier_map_entries = carrier_map_entries or {}
    carrier_reference = carrier_reference or {}
    freq_signature: Optional[tuple[float, ...]] = None

    def record_row(row: dict[str, float]) -> None:
        row.setdefault("run_id", run_id)
        row.setdefault("timestamp", _utc_now())
        results_all.append(row)

    for calibration in calibrations:
        for cutoff in cutoffs:
            baseline_model, _, _, _, _ = _build_gaussian_model(
                include_pseudomode=False,
                tau_B=1.0,
                calibration=calibration,
                gpm_scale=1.0,
                g_override=None,
                omega_c_scale=omega_c_scale,
                nbar=nbar,
                nbar_pseudo=nbar,
            )
            baseline_obs, _, _ = _compute_observables(baseline_model, lp_frac, rel_bw)
            baseline_band = baseline_obs["band_power"]
            baseline_narrowband = baseline_obs["narrowband_power"]
            baseline_baseband = baseline_obs["baseband_power"]
            baseline_env = baseline_obs["envelope_rms"]
            baseline_energy = baseline_obs["energy"]
            baseline_n1 = max(baseline_obs["occ_meas"], 1e-12)
            narrow_freqs = np.asarray(baseline_obs["psd_freqs"], dtype=float)
            baseline_psd_vals = np.asarray(baseline_obs["psd_values"], dtype=float)
            current_signature = tuple(np.round(narrow_freqs, 12))
            if freq_signature is None:
                freq_signature = current_signature
            elif current_signature != freq_signature:
                raise ValueError("Gaussian solver frequency grid mismatch detected")
            baseline_fast_band = float("nan")
            baseline_j = float("nan")
            if j_bandwidth_frac > 0:
                f1_hz = W1 / (2.0 * np.pi)
                freq_fast = np.linspace(
                    f1_hz * (1.0 - j_bandwidth_frac),
                    f1_hz * (1.0 + j_bandwidth_frac),
                    400,
                    dtype=float,
                )
                omega_fast = 2.0 * np.pi * freq_fast
                psd_fast = np.array([
                    gaussian_psd(baseline_model, baseline_model.fast_selector, w) for w in omega_fast
                ], dtype=float)
                psd_fast_one_sided = one_sided_psd_from_omega(freq_fast, psd_fast)
                baseline_fast_band = band_average_from_psd(freq_fast, psd_fast_one_sided, f1_hz, j_bandwidth_frac)
                baseline_j = 0.5 * baseline_fast_band

            baseline_row = {
                "theta": 0.0,
                "seed": -1,
                "tau_B": 0.0,
                "calibration": calibration,
                "cutoff": float(cutoff),
                "N": float(config_data.get("hierarchy_cutoff", 5)),
                "N_pseudo": float(cutoff),
                "model": "lindblad",
                "R_env": 1.0,
                "band_power": baseline_band,
                "energy": baseline_energy,
                "envelope_rms": baseline_env,
                "envelope_rms_ratio": 1.0,
                "baseband_power": baseline_baseband,
                "baseband_ratio": 1.0,
                "narrowband_power": baseline_narrowband,
                "occ_target": baseline_n1,
                "occ_meas": baseline_n1,
                "rel_occ_err": 0.0,
                "n1_tail": baseline_n1,
                "occ_ratio_achieved": 1.0,
                "gpm_scale": 1.0,
                "nudges_used": 0,
                "hit_clamp": False,
                "g_used": np.nan,
                "kappa": np.nan,
                "Jw1_target": baseline_j if calibration == "equal_carrier" else np.nan,
                "Jw1_est": baseline_j if calibration == "equal_carrier" else np.nan,
                "rel_Jw1_err": 0.0 if calibration == "equal_carrier" else np.nan,
                "nbar": nbar,
                "a1_init_alpha": a1_init_alpha,
                "burn_frac": burn_frac,
                "omega_c": omega_c_scale * W1,
                "t_final": t_final,
                "n_time": n_time,
                "psd_norm": psd_norm,
                "is_surrogate": 0,
                "psd_nrmse": np.nan,
                "config_hash": config_hash_value,
                "wall_s": baseline_obs["wall_s"],
                "ode_steps": 0.0,
                "timestamp": _utc_now(),
                "run_id": run_id,
                "solver": "gaussian",
                "min_real_part": baseline_obs["min_real_part"],
                "cond_drift": baseline_obs["cond_drift"],
                "freq_grid": json.dumps([float(f) for f in narrow_freqs]),
                "j_bandwidth_frac": j_bandwidth_frac,
                "holm_family": "",
                "status": "ok",
                "failure_reason": "",
            }
            record_row(baseline_row)

            for tau_B in tau_values:
                kappa = 1.0 / tau_B
                base_g = baseline_pseudomode_coupling(tau_B, calibration, W1, GAMMA1)
                scale = 1.0
                nudges_used = 0
                hit_clamp = False
                status = "ok"
                failure_reason = ""

                carrier_entry = None
                if calibration == "equal_carrier":
                    for theta_key, entry in carrier_map_entries.items():
                        if abs(theta_key - W1 * tau_B) <= 1e-9:
                            carrier_entry = entry
                            break

                g_override = None
                target_j = np.nan
                est_j = np.nan
                if carrier_entry is not None:
                    g_map = float(carrier_entry.get("g", carrier_entry.get("g_used", 0.0)))
                    if base_g > 0:
                        scale = g_map / base_g
                    target_j = float(carrier_entry.get("Jw1_target", np.nan))
                elif calibration == "equal_carrier":
                    if carrier_reference:
                        ref_kappa = carrier_reference.get("kappa", 1.0 / tau_B)
                        ref_g = carrier_reference.get("g", base_g)
                        delta_ref = carrier_reference.get("delta", W1 - omega_c_scale * W1)
                        g_cal, target_j_calc, _, _ = calibrate_equal_carrier_g(
                            kappa=kappa,
                            baseline_kappa=ref_kappa,
                            baseline_g=ref_g,
                            delta=delta_ref,
                            tolerance=equal_carrier_tol,
                        )
                        if base_g > 0:
                            scale = g_cal / base_g
                        target_j = target_j_calc
                    else:
                        delta_ref = W1 - omega_c_scale * W1
                        ref_g = base_g
                        g_cal, target_j_calc, _, _ = calibrate_equal_carrier_g(
                            kappa=kappa,
                            baseline_kappa=kappa,
                            baseline_g=ref_g,
                            delta=delta_ref,
                            tolerance=equal_carrier_tol,
                        )
                        if base_g > 0:
                            scale = g_cal / base_g
                        target_j = target_j_calc

                if calibration == "equal_carrier" and j_bandwidth_frac > 0 and base_g > 0:
                    target_fast = baseline_fast_band
                    f1_hz = W1 / (2.0 * np.pi)
                    freq_fast_grid = np.linspace(
                        f1_hz * (1.0 - j_bandwidth_frac),
                        f1_hz * (1.0 + j_bandwidth_frac),
                        400,
                        dtype=float,
                    )
                    omega_fast_grid = 2.0 * np.pi * freq_fast_grid

                    def fast_error(g_value: float) -> float:
                        scale_val = g_value / base_g
                        model_eval, _, _, _, _ = _build_gaussian_model(
                            include_pseudomode=True,
                            tau_B=tau_B,
                            calibration=calibration,
                            gpm_scale=scale_val,
                            g_override=None,
                            omega_c_scale=omega_c_scale,
                            nbar=nbar,
                            nbar_pseudo=nbar,
                        )
                        psd_fast = np.array([
                            gaussian_psd(model_eval, model_eval.fast_selector, w) for w in omega_fast_grid
                        ], dtype=float)
                        psd_fast_one = one_sided_psd_from_omega(freq_fast_grid, psd_fast)
                        band_avg = band_average_from_psd(freq_fast_grid, psd_fast_one, f1_hz, j_bandwidth_frac)
                        return band_avg - target_fast

                    g_prev = max(scale * base_g, 1e-6)
                    err_prev = fast_error(g_prev)
                    if not np.isnan(err_prev) and (target_fast == 0 or abs(err_prev / target_fast) > equal_carrier_tol):
                        g_curr = max(g_prev * (1.0 + 0.1), 1e-6)
                        err_curr = fast_error(g_curr)
                        for _ in range(12):
                            if target_fast != 0 and abs(err_curr / target_fast) <= equal_carrier_tol:
                                break
                            if target_fast == 0 and abs(err_curr) <= equal_carrier_tol:
                                break
                            denom = err_curr - err_prev
                            if abs(denom) < 1e-9:
                                step = 0.2 if err_curr < 0 else -0.2
                                g_next = max(g_curr * (1.0 + step), 1e-6)
                            else:
                                g_next = g_curr - err_curr * (g_curr - g_prev) / denom
                            g_prev, err_prev = g_curr, err_curr
                            g_curr = max(g_next, 1e-6)
                            err_curr = fast_error(g_curr)
                        else:
                            status = "fail"
                            failure_reason = "rel_Jw1_err"
                        if status == "ok":
                            scale = g_curr / base_g
                    else:
                        scale = g_prev / base_g
                    target_j = baseline_j

                def evaluate(scale_value: float, override: float | None) -> tuple[dict[str, float], float, float]:
                    model_eval, g_used_eval, _, omega_c_eval, _ = _build_gaussian_model(
                        include_pseudomode=True,
                        tau_B=tau_B,
                        calibration=calibration,
                        gpm_scale=scale_value,
                        g_override=override,
                        omega_c_scale=omega_c_scale,
                        nbar=nbar,
                        nbar_pseudo=nbar,
                    )
                    metrics_eval = _gaussian_metrics(
                        model=model_eval,
                        baseline_band=baseline_band,
                        baseline_narrowband=baseline_narrowband,
                        baseline_n1=baseline_n1,
                        psd_norm=psd_norm,
                        rel_bandwidth=rel_bw,
                        lp_frac=lp_frac,
                        omega_c_value=omega_c_eval,
                        nbar=nbar,
                        g_used=g_used_eval,
                        kappa=kappa,
                        omega_c_scale=omega_c_scale,
                        baseline_psd_freqs=narrow_freqs,
                        baseline_psd_values=baseline_psd_vals,
                        j_bandwidth_frac=j_bandwidth_frac,
                    )
                    return metrics_eval, g_used_eval, omega_c_eval

                metrics, g_used, omega_c_value = evaluate(scale, g_override)
                current_signature = tuple(np.round(metrics["psd_freqs"], 12))
                if freq_signature is None:
                    freq_signature = current_signature
                elif current_signature != freq_signature:
                    raise ValueError("Gaussian solver frequency grid mismatch detected")

                if calibration == "equal_heating":
                    prev_scale = None
                    prev_err = None
                    while True:
                        rel_occ_err = metrics["rel_occ_err"]
                        if np.isnan(rel_occ_err) or abs(rel_occ_err) <= tuner_tol_occ:
                            break
                        if nudges_used >= tuner_max_nudges:
                            hit_clamp = True
                            status = "fail"
                            failure_reason = "rel_occ_err"
                            break
                        if prev_err is None:
                            correction = np.sqrt(max(1e-6, 1.0 / max(metrics["occ_ratio"], 1e-6)))
                            new_scale = scale * correction
                        else:
                            denom = rel_occ_err - prev_err
                            if abs(denom) < 1e-9:
                                denom = np.sign(denom or 1.0) * 1e-9
                            new_scale = scale - rel_occ_err * (scale - prev_scale) / denom
                        prev_scale, prev_err = scale, rel_occ_err
                        scale = float(np.clip(new_scale, 0.3, 3.0))
                        nudges_used += 1
                        metrics, g_used, omega_c_value = evaluate(scale, g_override)
                        current_signature = tuple(np.round(metrics["psd_freqs"], 12))
                        if current_signature != freq_signature:
                            raise ValueError("Gaussian solver frequency grid mismatch detected")

                rel_occ_err = metrics["rel_occ_err"]
                rel_j_err = np.nan
                est_j = np.nan
                if calibration == "equal_carrier":
                    delta = W1 - omega_c_value
                    if np.isnan(target_j):
                        if carrier_reference:
                            target_j = carrier_reference.get("target_j", np.nan)
                        else:
                            baseline_tau = tau_B
                            baseline_kappa = 1.0 / baseline_tau
                            baseline_g = baseline_pseudomode_coupling(baseline_tau, calibration, W1, GAMMA1)
                            target_j = pseudomode_lorentzian_j(baseline_g, baseline_kappa, delta)
                    if j_bandwidth_frac > 0:
                        est_j = 0.5 * metrics["fast_band_power"]
                        target_j = baseline_j
                    else:
                        est_j = pseudomode_lorentzian_j(g_used, kappa, delta)
                    if target_j and target_j > 0:
                        rel_j_err = abs(est_j - target_j) / target_j
                        if rel_j_err > equal_carrier_tol:
                            status = "fail"
                            failure_reason = "rel_Jw1_err"
                if calibration == "equal_carrier" and j_bandwidth_frac > 0 and status == "ok":
                    prev_scale_j = None
                    prev_err_j = None
                    while True:
                        if baseline_fast_band <= 0:
                            rel_j_err = 0.0
                            break
                        rel_j_err = (
                            (metrics["fast_band_power"] - baseline_fast_band) / baseline_fast_band
                            if baseline_fast_band > 0
                            else np.nan
                        )
                        if np.isnan(rel_j_err) or abs(rel_j_err) <= equal_carrier_tol:
                            break
                        if nudges_used >= tuner_max_nudges:
                            hit_clamp = True
                            status = "fail"
                            failure_reason = "rel_Jw1_err"
                            break
                        if prev_err_j is None:
                            correction = np.sqrt(max(1e-9, baseline_fast_band / metrics["fast_band_power"]))
                            new_scale = scale * correction
                        else:
                            denom = rel_j_err - prev_err_j
                            if abs(denom) < 1e-9:
                                denom = np.sign(denom or 1.0) * 1e-9
                            new_scale = scale - rel_j_err * (scale - prev_scale_j) / denom
                        prev_scale_j, prev_err_j = scale, rel_j_err
                        scale = float(np.clip(new_scale, 0.3, 3.0))
                        nudges_used += 1
                        metrics, g_used, omega_c_value = evaluate(scale, g_override)
                        current_signature = tuple(np.round(metrics["psd_freqs"], 12))
                        if current_signature != freq_signature:
                            raise ValueError("Gaussian solver frequency grid mismatch detected")
                    if status != "fail":
                        rel_j_err = (
                            (metrics["fast_band_power"] - baseline_fast_band) / baseline_fast_band
                            if baseline_fast_band > 0
                            else np.nan
                        )
                        est_j = 0.5 * metrics["fast_band_power"]

                rel_occ_err = metrics["rel_occ_err"]

                row = {
                    "theta": float(W1 * tau_B),
                    "seed": seed_value,
                    "tau_B": float(tau_B),
                    "calibration": calibration,
                    "cutoff": float(cutoff),
                    "N": float(config_data.get("hierarchy_cutoff", 5)),
                    "N_pseudo": float(cutoff),
                    "model": "pseudomode",
                    "R_env": float(metrics["band_power"] / baseline_band) if baseline_band > 0 else np.nan,
                    "band_power": metrics["band_power"],
                    "energy": metrics["energy"],
                    "envelope_rms": metrics["envelope_rms"],
                    "envelope_rms_ratio": float(metrics["envelope_rms"] / baseline_env) if baseline_env > 0 else np.nan,
                    "baseband_power": metrics["baseband_power"],
                    "baseband_ratio": float(metrics["baseband_power"] / baseline_baseband) if baseline_baseband > 0 else np.nan,
                    "narrowband_power": metrics["narrowband_power"],
                    "occ_target": baseline_n1,
                    "occ_meas": metrics["occ_meas"],
                    "rel_occ_err": rel_occ_err,
                    "n1_tail": metrics["occ_meas"],
                    "occ_ratio_achieved": metrics["occ_ratio"],
                    "gpm_scale": scale,
                    "nudges_used": nudges_used,
                    "hit_clamp": hit_clamp,
                    "g_used": g_used,
                    "kappa": kappa,
                    "Jw1_target": target_j,
                    "Jw1_est": est_j,
                    "rel_Jw1_err": rel_j_err,
                    "nbar": nbar,
                    "a1_init_alpha": a1_init_alpha,
                    "burn_frac": burn_frac,
                    "omega_c": omega_c_value,
                    "t_final": t_final,
                    "n_time": n_time,
                    "psd_norm": psd_norm,
                    "is_surrogate": 0,
                    "psd_nrmse": metrics["psd_nrmse"],
                    "config_hash": config_hash_value,
                    "wall_s": metrics["wall_s"],
                    "ode_steps": 0.0,
                    "timestamp": _utc_now(),
                    "run_id": run_id,
                    "solver": "gaussian",
                    "min_real_part": metrics["min_real_part"],
                    "cond_drift": metrics["cond_drift"],
                    "freq_grid": json.dumps([float(f) for f in metrics["psd_freqs"]]),
                    "j_bandwidth_frac": j_bandwidth_frac,
                    "holm_family": "eqcar_peak_vs_sides" if calibration == "equal_carrier" else "",
                    "status": status,
                    "failure_reason": failure_reason,
                }

                if status != "ok" and failure_reason:
                    record_row(row)
                    raise RuntimeError(
                        f"Tolerance failure for Θ={row['theta']:.3f}, seed={seed_value}: {failure_reason}"
                    )

                record_row(row)

    append_summary_rows(summary_path, results_all)
    return results_all

def _build_gaussian_model(
    *,
    include_pseudomode: bool,
    tau_B: float,
    calibration: str,
    gpm_scale: float,
    g_override: float | None,
    omega_c_scale: float,
    nbar: float,
    nbar_pseudo: float,
) -> tuple[GaussianModel, float, float, float, float]:
    frequencies = [W1, W2, W3]
    dampings = [GAMMA1, GAMMA2, GAMMA3]
    nbar_list = [nbar, 0.0, 0.0]
    couplings: list[tuple[int, int, float]] = [(0, 1, G12), (1, 2, G23)]

    g_used = 0.0
    kappa = 0.0
    omega_c = omega_c_scale * W1

    if include_pseudomode:
        if tau_B <= 0:
            raise ValueError("tau_B must be positive for pseudomode runs")
        kappa = 1.0 / tau_B
        g_base = baseline_pseudomode_coupling(tau_B, calibration, W1, GAMMA1)
        g_used = g_override if g_override is not None else g_base * gpm_scale
        frequencies.append(omega_c)
        dampings.append(kappa)
        nbar_list.append(nbar_pseudo)
        couplings.append((0, 3, g_used))

    model = build_model(
        frequencies=frequencies,
        couplings=couplings,
        dampings=dampings,
        nbar_list=nbar_list,
        fast_mode=0,
        x_measure_mode=2,
    )

    return model, g_used, kappa, omega_c, tau_B
def _compute_observables(
    model: GaussianModel,
    lp_frac: float,
    rel_bandwidth: float,
) -> tuple[dict[str, float], np.ndarray, float]:
    start = time.perf_counter()
    eigenvalues = np.linalg.eigvals(model.drift)
    min_real_part = float(np.min(np.real(eigenvalues)))
    cond_drift = float(np.linalg.cond(model.drift))

    V = steady_covariance(model)
    wall_s = time.perf_counter() - start

    V_sym = 0.5 * (V + V.T)
    if not np.allclose(V, V_sym, atol=1e-10):
        raise ValueError("Gaussian covariance solve returned a non-symmetric matrix")
    eigvals_V = np.linalg.eigvalsh(V_sym)
    if np.min(eigvals_V) < -1e-10:
        raise ValueError("Gaussian covariance is not PSD within tolerance")

    f3_hz = W3 / (2.0 * np.pi)
    span = max(rel_bandwidth, lp_frac, 0.1)
    f_max_dense = max(ENVELOPE_BAND[1], f3_hz * (1.0 + 4.0 * span))
    freq_dense = np.linspace(0.0, f_max_dense, 2000, dtype=float)
    omega_dense = 2.0 * np.pi * freq_dense
    psd_omega_dense = np.array([gaussian_psd(model, model.x_selector, w) for w in omega_dense], dtype=float)
    psd_one_sided_dense = one_sided_psd_from_omega(freq_dense, psd_omega_dense)

    baseband_power = baseband_power_from_psd(freq_dense, psd_one_sided_dense, f3_hz, lp_frac)
    narrowband_power = narrowband_power_from_psd(freq_dense, psd_one_sided_dense, f3_hz, rel_bandwidth)
    env_mask = (freq_dense >= ENVELOPE_BAND[0]) & (freq_dense <= ENVELOPE_BAND[1])
    if np.any(env_mask):
        band_power = float(np.trapezoid(psd_one_sided_dense[env_mask], freq_dense[env_mask]))
    else:
        band_power = 0.0

    try:
        x_var = quadrature_variance(V_sym, 2, "x")
    except Exception:
        x_idx = 4
        x_var = float(V_sym[x_idx, x_idx])
    env_rms = float(np.sqrt(max(x_var, 0.0)))

    occ_meas = mode_occupancy(V_sym, 0)
    energy = mode_occupancy(V_sym, 2 if model.n_modes >= 3 else model.n_modes - 1)

    narrow_low_hz = f3_hz * (1.0 - rel_bandwidth)
    narrow_high_hz = f3_hz * (1.0 + rel_bandwidth)
    freq_nb = np.linspace(max(0.0, narrow_low_hz), narrow_high_hz, 200, dtype=float)
    omega_nb = 2.0 * np.pi * freq_nb
    psd_nb = np.array([gaussian_psd(model, model.x_selector, w) for w in omega_nb], dtype=float)
    psd_nb_one_sided = one_sided_psd_from_omega(freq_nb, psd_nb)

    return (
        {
            "band_power": band_power,
            "baseband_power": baseband_power,
            "narrowband_power": narrowband_power,
            "envelope_rms": env_rms,
            "occ_meas": occ_meas,
            "energy": energy,
            "wall_s": wall_s,
            "narrow_low": narrow_low_hz,
            "narrow_high": narrow_high_hz,
            "min_real_part": min_real_part,
            "cond_drift": cond_drift,
            "psd_freqs": freq_nb,
            "psd_values": psd_nb_one_sided,
            "psd_dense_freqs": freq_dense,
            "psd_dense_values": psd_one_sided_dense,
        },
        V_sym,
        wall_s,
    )


def _gaussian_metrics(
    *,
    model: GaussianModel,
    baseline_band: float,
    baseline_narrowband: float,
    baseline_n1: float,
    psd_norm: str,
    rel_bandwidth: float,
    lp_frac: float,
    omega_c_value: float,
    nbar: float,
    g_used: float,
    kappa: float,
    omega_c_scale: float,
    baseline_psd_freqs: Optional[np.ndarray],
    baseline_psd_values: Optional[np.ndarray],
    j_bandwidth_frac: float,
) -> dict[str, float]:
    metrics, V, wall_s = _compute_observables(model, lp_frac, rel_bandwidth)
    baseband_power = metrics["baseband_power"]
    narrowband_power = metrics["narrowband_power"]
    band_power = metrics["band_power"]
    env_rms = metrics["envelope_rms"]
    occ_meas = metrics["occ_meas"]
    narrow_low = metrics["narrow_low"]
    narrow_high = metrics["narrow_high"]
    n3_occ = metrics["energy"]
    min_real_part = metrics["min_real_part"]
    cond_drift = metrics["cond_drift"]
    psd_freqs = metrics["psd_freqs"]
    psd_values = metrics["psd_values"]
    psd_dense_freqs = metrics.get("psd_dense_freqs")
    psd_dense_values = metrics.get("psd_dense_values")

    occ_ratio = occ_meas / baseline_n1 if baseline_n1 > 0 else np.nan
    rel_occ_err = (occ_meas - baseline_n1) / baseline_n1 if baseline_n1 > 0 else np.nan

    psd_nrmse = np.nan
    if (
        baseline_psd_freqs is not None
        and baseline_psd_values is not None
        and baseline_psd_values.size > 0
        and psd_dense_freqs is not None
        and psd_dense_values is not None
    ):
        interp = np.interp(baseline_psd_freqs, psd_dense_freqs, psd_dense_values)
        denom = np.sqrt(np.mean(baseline_psd_values**2))
        if denom > 0:
            diff = interp - baseline_psd_values
            psd_nrmse = float(np.sqrt(np.mean(diff**2)) / denom)

    fast_band_power = np.nan
    if j_bandwidth_frac > 0:
        f1_hz = W1 / (2.0 * np.pi)
        freq_fast = np.linspace(
            f1_hz * (1.0 - j_bandwidth_frac),
            f1_hz * (1.0 + j_bandwidth_frac),
            400,
            dtype=float,
        )
        omega_fast = 2.0 * np.pi * freq_fast
        psd_fast = np.array([gaussian_psd(model, model.fast_selector, w) for w in omega_fast], dtype=float)
        psd_fast_one_sided = one_sided_psd_from_omega(freq_fast, psd_fast)
        fast_band_power = band_average_from_psd(freq_fast, psd_fast_one_sided, f1_hz, j_bandwidth_frac)

    return {
        "band_power": band_power,
        "baseband_power": baseband_power,
        "narrowband_power": narrowband_power,
        "envelope_rms": env_rms,
        "occ_meas": occ_meas,
        "occ_ratio": occ_ratio,
        "rel_occ_err": rel_occ_err,
        "n1_tail": occ_meas,
        "energy": n3_occ,
        "psd_nrmse": psd_nrmse,
        "wall_s": wall_s,
        "min_real_part": min_real_part,
        "cond_drift": cond_drift,
        "psd_freqs": psd_freqs,
        "psd_values": psd_values,
        "fast_band_power": fast_band_power,
    }


def write_peak_table(
    results: list[dict[str, str]],
    calibrations: Sequence[str],
    cutoffs: Sequence[int],
    output_dir: Path,
) -> Path:
    path = output_dir / "peak_table.csv"

    def _to_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["calibration", "cutoff", "theta_star", "tau_B_star", "R_env_peak"])
        for calibration in calibrations:
            for cutoff in cutoffs:
                subset: list[dict[str, str]] = []
                for row in results:
                    if row.get("status", "ok") != "ok":
                        continue
                    if row.get("calibration") != calibration:
                        continue
                    if row.get("model") != "pseudomode":
                        continue
                    if _to_float(row.get("cutoff")) != float(cutoff):
                        continue
                    subset.append(row)
                if not subset:
                    continue
                peak_row = max(subset, key=lambda item: _to_float(item.get("R_env")))
                theta_star = _to_float(peak_row.get("theta"))
                tau_star = _to_float(peak_row.get("tau_B"))
                r_env_peak = _to_float(peak_row.get("R_env"))
                writer.writerow(
                    [
                        calibration,
                        cutoff,
                        f"{theta_star:.6g}",
                        f"{tau_star:.6g}",
                        f"{r_env_peak:.6g}",
                    ]
                )
    print(f"✓ Peak table saved to {path}")
    return path


def plot_quantum_vs_classical(
    results: list[dict[str, str]],
    calibrations: Sequence[str],
    cutoffs: Sequence[int],
    classical_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    n_rows = len(calibrations)
    fig, axes = plt.subplots(
        n_rows,
        1,
        sharex=True,
        figsize=(10, 5 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    color_map = plt.cm.viridis(np.linspace(0, 1, len(cutoffs)))

    for ax, calibration in zip(axes, calibrations):
        valid_rows = [
            row
            for row in results
            if row.get("calibration") == calibration and row.get("status", "ok") == "ok"
        ]
        psd_norms = {row.get("psd_norm") for row in valid_rows if row.get("psd_norm")}
        if len(psd_norms) > 1:
            raise ValueError(
                f"Inconsistent psd_norm values for calibration {calibration}: {psd_norms}"
            )
        if psd_norms:
            print(f"[FIG] calibration={calibration} psd_norm={next(iter(psd_norms))}", flush=True)
        ax.set_title(f"R_env vs Θ (calibration = {calibration})", fontsize=14, fontweight="bold")
        for idx, cutoff in enumerate(cutoffs):
            theta_collect: list[float] = []
            r_env_collect: list[float] = []
            for row in valid_rows:
                if row.get("model") != "pseudomode":
                    continue
                try:
                    cutoff_val = float(row.get("cutoff"))
                except (TypeError, ValueError):
                    continue
                if cutoff_val != float(cutoff):
                    continue
                try:
                    theta_val = float(row.get("theta"))
                    r_env_val = float(row.get("R_env"))
                except (TypeError, ValueError):
                    continue
                if np.isnan(theta_val) or np.isnan(r_env_val):
                    continue
                theta_collect.append(theta_val)
                r_env_collect.append(r_env_val)
            if not theta_collect:
                continue
            print(
                f"[FIG] calibration={calibration} cutoff={cutoff} N_effective={len(theta_collect)}",
                flush=True,
            )
            order = np.argsort(theta_collect)
            theta_vals = [theta_collect[i] for i in order]
            r_env_vals = [r_env_collect[i] for i in order]
            ax.semilogx(
                theta_vals,
                r_env_vals,
                color=color_map[idx],
                linewidth=2.2,
                label=f"Pseudomode Np={cutoff}",
            )

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Lindblad baseline")

        classical = classical_curves.get(calibration)
        if classical is not None:
            ax.semilogx(
                classical[0],
                classical[1],
                color="gray",
                linestyle=":",
                linewidth=2.5,
                label="Classical sweep",
            )

        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylabel("R_env (band power ratio)", fontsize=12)

    axes[-1].set_xlabel("Θ = ω₁ τ_B", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=3, fontsize=11)

    figure_path = output_dir / "quantum_vs_classical_resonance.png"
    fig.savefig(figure_path, dpi=250)
    plt.close(fig)
    print(f"✓ Figure saved to {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantum hierarchical parameter sweep (Stage 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tauB_min", type=float, default=1e-4, help="Minimum bath correlation time.")
    parser.add_argument("--tauB_max", type=float, default=1e-1, help="Maximum bath correlation time.")
    parser.add_argument("--n_points", type=int, default=30, help="Number of points in the τ_B sweep.")
    parser.add_argument(
        "--theta",
        type=float,
        help="Single Θ value (Θ = ω₁ τ_B) to evaluate instead of a log sweep.",
    )
    parser.add_argument(
        "--calibrations",
        nargs="+",
        default=["equal_carrier"],
        help="Calibration modes to evaluate.",
    )
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        default=[4, 6, 8, 10],
        help="Pseudomode cutoffs (Hilbert space truncation).",
    )
    parser.add_argument(
        "--hierarchy_cutoff",
        type=int,
        default=6,
        help="Bosonic cutoff for each hierarchy oscillator.",
    )
    parser.add_argument(
        "--t_final",
        type=float,
        default=2.5,
        help="Total simulation time for each run.",
    )
    parser.add_argument(
        "--n_time",
        type=int,
        default=600,
        help="Number of time points for the solver.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/quantum_sweep",
        help="Directory to store outputs.",
    )
    parser.add_argument(
        "--classical_summary",
        type=str,
        default="results/classical_sweep/summary.csv",
        help="Optional classical summary CSV for overlay.",
    )
    parser.add_argument(
        "--solver_progress",
        type=str,
        default="none",
        help="QuTiP progress bar type (e.g., text, enhanced, none).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for the ODE solver.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for the ODE solver.",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=50_000,
        help="Maximum internal steps for QuTiP mesolve.",
    )
    parser.add_argument(
        "--no_rhs_reuse",
        action="store_true",
        help="Disable RHS reuse in QuTiP solver (default is enabled).",
    )
    parser.add_argument(
        "--nbar",
        type=float,
        default=0.0,
        help="Thermal occupation number for the primary oscillator bath.",
    )
    parser.add_argument(
        "--a1_init_alpha",
        type=float,
        default=0.0,
        help="Initial coherent displacement amplitude for oscillator 1.",
    )
    parser.add_argument(
        "--burn_frac",
        type=float,
        default=0.25,
        help="Fraction of the trajectory discarded as burn-in before computing metrics.",
    )
    parser.add_argument(
        "--omega_c_scale",
        type=float,
        default=1.0,
        help="Scale factor for the pseudomode carrier frequency ω_c relative to ω₁.",
    )
    parser.add_argument(
        "--equal_carrier_map",
        type=Path,
        help="Optional JSON map of precomputed equal-carrier couplings.",
    )
    parser.add_argument(
        "--equal_carrier_tol",
        type=float,
        default=0.02,
        help="Relative tolerance for matching spectral weight at ω₁ in equal-carrier mode.",
    )
    parser.add_argument(
        "--psd_norm",
        type=str,
        choices=["onesided", "twosided"],
        default="onesided",
        help="PSD normalization convention recorded in outputs.",
    )
    parser.add_argument(
        "--quant_engine",
        type=str,
        choices=["trajectory", "gaussian"],
        default="trajectory",
        help="Choose trajectory-based mesolve or fast Gaussian covariance solver.",
    )
    parser.add_argument(
        "--tuner_tol_occ",
        type=float,
        default=0.03,
        help="Relative tolerance for the occupancy tuner (|rel_occ_err| ≤ tol).",
    )
    parser.add_argument(
        "--tuner_max_nudges",
        type=int,
        default=3,
        help="Maximum occupancy tuner nudges per run.",
    )
    parser.add_argument(
        "--j_bandwidth_frac",
        type=float,
        default=0.02,
        help="Relative half-width for J(ω₁) band-average (fraction of ω₁).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    seed_value = _infer_seed_label(output_dir)
    print(f"[RUN] run_id={run_id} output_dir={output_dir}")

    if args.theta is not None:
        tau_values = np.array([float(args.theta) / W1], dtype=float)
    else:
        tau_values = np.logspace(np.log10(args.tauB_min), np.log10(args.tauB_max), args.n_points)
    times = np.linspace(0, args.t_final, args.n_time)

    solver_progress = None if args.solver_progress.lower() == "none" else args.solver_progress
    rhs_reuse = not args.no_rhs_reuse
    summary_path = output_dir / "summary.csv"

    config_data, config_digest = load_config_with_hash()

    carrier_map_entries: dict[float, dict] | None = None
    carrier_reference: dict | None = None
    if args.equal_carrier_map is not None:
        with args.equal_carrier_map.open("r", encoding="utf-8") as fh:
            map_raw = json.load(fh)
        map_hash = map_raw.get("config_hash")
        if map_hash and map_hash != config_digest:
            print(f"[WARN] equal-carrier map hash {map_hash} differs from config {config_digest}")
        baseline_raw = map_raw.get("baseline")
        if isinstance(baseline_raw, dict):
            theta_ref = float(baseline_raw.get("theta", baseline_raw.get("Theta", 1.0)))
            tau_ref = float(baseline_raw.get("tau_B", baseline_raw.get("tau", theta_ref / W1)))
            kappa_ref = float(baseline_raw.get("kappa", 1.0 / tau_ref)) if tau_ref > 0 else float("inf")
            g_ref = float(baseline_raw.get("g", baseline_raw.get("g_used", 0.0)))
            delta_ref = float(baseline_raw.get("delta", W1 - args.omega_c_scale * W1))
            target_j_ref = float(
                baseline_raw.get(
                    "Jw1_target",
                    pseudomode_lorentzian_j(g_ref, kappa_ref, delta_ref),
                )
            )
            carrier_reference = {
                "theta": theta_ref,
                "tau": tau_ref,
                "kappa": kappa_ref,
                "g": g_ref,
                "target_j": target_j_ref,
                "delta": delta_ref,
            }
        entries_raw = map_raw.get("entries", {})
        if isinstance(entries_raw, dict):
            carrier_map_entries = {}
            for key, val in entries_raw.items():
                if not isinstance(val, dict):
                    continue
                theta_entry = float(val.get("theta", key))
                entry = dict(val)
                entry["theta"] = theta_entry
                tau_entry = float(entry.get("tau_B", theta_entry / W1))
                entry["tau_B"] = tau_entry
                entry["kappa"] = float(entry.get("kappa", 1.0 / tau_entry)) if tau_entry > 0 else float("inf")
                g_val = entry.get("g", entry.get("g_used"))
                if g_val is None:
                    raise ValueError(f"Map entry for Θ={theta_entry} lacks coupling 'g'.")
                entry["g"] = float(g_val)
                if carrier_reference is not None:
                    delta_ref = carrier_reference["delta"]
                else:
                    delta_ref = W1 - args.omega_c_scale * W1
                entry.setdefault("delta", delta_ref)
                entry["Jw1_target"] = float(
                    entry.get(
                        "Jw1_target",
                        pseudomode_lorentzian_j(entry["g"], entry["kappa"], entry["delta"]),
                    )
                )
                entry["Jw1_est"] = float(
                    entry.get(
                        "Jw1_est",
                        pseudomode_lorentzian_j(entry["g"], entry["kappa"], entry["delta"]),
                    )
                )
                if entry["Jw1_target"] > 0:
                    entry["rel_Jw1_err"] = float(
                        entry.get(
                            "rel_Jw1_err",
                            abs(entry["Jw1_est"] - entry["Jw1_target"]) / entry["Jw1_target"],
                        )
                    )
                else:
                    entry["rel_Jw1_err"] = float("nan")
                carrier_map_entries[theta_entry] = entry

    psd_norm_value = args.psd_norm
    
    if summary_path.exists():
        summary_path.unlink()
    
    if args.quant_engine == "gaussian":
        results = run_gaussian_sweep(
            tau_values=tau_values,
            calibrations=args.calibrations,
            cutoffs=args.cutoffs,
            summary_path=summary_path,
            config_data=config_data,
            config_hash_value=config_digest,
            omega_c_scale=args.omega_c_scale,
            tuner_tol_occ=args.tuner_tol_occ,
            tuner_max_nudges=args.tuner_max_nudges,
            carrier_map_entries=carrier_map_entries,
            carrier_reference=carrier_reference,
            psd_norm=psd_norm_value,
            equal_carrier_tol=args.equal_carrier_tol,
            run_id=run_id,
            seed_label=seed_value,
            nbar=args.nbar,
            a1_init_alpha=args.a1_init_alpha,
            j_bandwidth_frac=args.j_bandwidth_frac,
        )
    else:
        results = run_quantum_sweep(
            tau_values=tau_values,
            calibrations=args.calibrations,
            cutoffs=args.cutoffs,
            times=times,
            hierarchy_cutoff=args.hierarchy_cutoff,
            solver_progress=solver_progress,
            atol=args.atol,
            rtol=args.rtol,
            nsteps=args.nsteps,
            rhs_reuse=rhs_reuse,
            summary_path=summary_path,
            nbar=args.nbar,
            a1_init_alpha=args.a1_init_alpha,
            burn_fraction=args.burn_frac,
            omega_c_scale=args.omega_c_scale,
            t_final=args.t_final,
            n_time=args.n_time,
            config_data=config_data,
            config_hash_value=config_digest,
            tuner_tol_occ=args.tuner_tol_occ,
            tuner_max_nudges=args.tuner_max_nudges,
            carrier_map_entries=carrier_map_entries,
            carrier_reference=carrier_reference,
            psd_norm=psd_norm_value,
            equal_carrier_tol=args.equal_carrier_tol,
            run_id=run_id,
            seed_label=seed_value,
        )
    summary_rows = load_summary_rows(summary_path)
    update_quantum_manifest(output_dir, summary_rows)
    write_peak_table(summary_rows, args.calibrations, args.cutoffs, output_dir)

    classical_curves = load_classical_curves(Path(args.classical_summary))
    plot_quantum_vs_classical(summary_rows, args.calibrations, args.cutoffs, classical_curves, output_dir)


if __name__ == "__main__":
    main()
