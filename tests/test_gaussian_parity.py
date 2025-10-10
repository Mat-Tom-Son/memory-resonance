import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

THETA = 0.95
ABS_TOL = 1e-8
REL_TOL = 1e-3
ENV_LIMIT_THREADS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _run_stage3(tmpdir: Path, engine: str) -> Path:
    """
    Execute stage3_parameter_sweep.py for a single Θ using the requested engine.
    Returns the directory containing summary.csv for that run.
    """
    out_dir = tmpdir / engine
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "stage3_parameter_sweep.py",
        "--theta",
        f"{THETA}",
        "--calibrations",
        "equal_carrier",
        "--equal_carrier_map",
        "calibration/equal_carrier_map.json",
        "--quant_engine",
        engine,
        "--psd_norm",
        "onesided",
        "--cutoffs",
        "4",
        "--hierarchy_cutoff",
        "4",
        "--t_final",
        "2.0",
        "--n_time",
        "200",
        "--output_dir",
        str(out_dir),
        "--solver_progress",
        "none",
        "--j_bandwidth_frac",
        "0",
    ]
    env = os.environ.copy()
    env.update(ENV_LIMIT_THREADS)
    subprocess.run(cmd, check=True, env=env, cwd=str(Path(__file__).resolve().parents[1]))
    summary_path = out_dir / "summary.csv"
    if not summary_path.exists():
        raise AssertionError(f"summary.csv not produced for engine={engine}")
    return summary_path


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def _select_row(rows: list[dict[str, str]]) -> dict[str, str]:
    candidates = []
    for row in rows:
        try:
            theta_val = float(row.get("theta", "nan"))
        except ValueError:
            continue
        if math.isclose(theta_val, THETA, rel_tol=1e-9, abs_tol=1e-12) and row.get("model") == "pseudomode":
            candidates.append(row)
    if not candidates:
        raise AssertionError("No pseudomode rows found at Θ=0.95")
    # sort by timestamp to pick latest entry
    candidates.sort(key=lambda r: r.get("timestamp", ""))
    return candidates[-1]


def _parse_float(row: dict[str, str], field: str) -> float:
    value = row.get(field)
    if value is None or value == "":
        raise AssertionError(f"Missing field '{field}' in row")
    try:
        result = float(value)
    except ValueError as exc:
        raise AssertionError(f"Field '{field}' is not numeric: {value}") from exc
    return result


def _rel_close(a: float, b: float, *, rel: float = REL_TOL, abs_tol: float = ABS_TOL) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


def test_parity_gaussian_vs_trajectory():
    assert os.path.exists("calibration/equal_carrier_map.json"), "Missing equal-carrier calibration map"

    tmp_root = Path(tempfile.mkdtemp(prefix="parity_")).resolve()
    try:
        traj_summary = _run_stage3(tmp_root, "trajectory")
        gauss_summary = _run_stage3(tmp_root, "gaussian")

        traj_row = _select_row(_read_summary(traj_summary))
        gauss_row = _select_row(_read_summary(gauss_summary))

        for field in ("baseband_power", "baseband_ratio", "narrowband_power", "occ_meas"):
            t_val = _parse_float(traj_row, field)
            g_val = _parse_float(gauss_row, field)
            assert _rel_close(t_val, g_val), f"{field} mismatch: {t_val} vs {g_val}"

        if gauss_row.get("Jw1_est") not in (None, "", "nan"):
            t_j = _parse_float(traj_row, "Jw1_est")
            g_j = _parse_float(gauss_row, "Jw1_est")
            assert _rel_close(t_j, g_j), f"Jw1_est mismatch: {t_j} vs {g_j}"

        if gauss_row.get("rel_Jw1_err") not in (None, "", "nan"):
            t_err = _parse_float(traj_row, "rel_Jw1_err")
            g_err = _parse_float(gauss_row, "rel_Jw1_err")
            assert _rel_close(t_err, g_err), f"rel_Jw1_err mismatch: {t_err} vs {g_err}"

        min_real_part = _parse_float(gauss_row, "min_real_part")
        assert min_real_part < -1e-6, f"Gaussian drift marginally stable: min_real_part={min_real_part}"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
