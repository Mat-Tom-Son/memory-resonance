"""
Compute operational S/C/M simplex scores from existing CSVs.

S: spectral overlap -> 1 - clamp(PSD-NRMSE)
C: coherence/modulation -> normalized envelope delta between OU and PSD surrogate
M: equal-carrier enhancement -> normalized lift gated by |ΔJ|/J* tolerance
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


@dataclass
class SimplexResult:
    S: float
    C: float
    M: float
    S_sem: float
    C_sem: float
    M_sem: float

    def barycentric(self) -> tuple[float, float, float]:
        s, c, m = max(self.S, 0.0), max(self.C, 0.0), max(self.M, 0.0)
        t = s + c + m
        if t <= 0:
            return (0.0, 0.0, 0.0)
        return (s / t, c / t, m / t)


def compute_S_from_psd_nrmse(csv_classical: Path) -> Tuple[float, float]:
    df = pd.read_csv(csv_classical)
    if "psd_nrmse" not in df.columns:
        raise ValueError("CSV must contain psd_nrmse column")
    vals = np.asarray(df["psd_nrmse"].values, dtype=float)
    svals = 1.0 - np.clip(vals, 0.0, 1.0)
    return float(np.nanmean(svals)), float(np.nanstd(svals, ddof=1) / np.sqrt(max(1, np.isfinite(svals).sum())))


def compute_C_from_band_delta(csv_classical: Path) -> Tuple[float, float]:
    df = pd.read_csv(csv_classical)
    required = {"band_mean", "band_mean_surrogate"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain band_mean and band_mean_surrogate")
    a = np.asarray(df["band_mean"].values, dtype=float)
    b = np.asarray(df["band_mean_surrogate"].values, dtype=float)
    # Normalized difference
    with np.errstate(divide="ignore", invalid="ignore"):
        c = (a - b) / np.maximum(1e-12, a)
    return float(np.nanmean(c)), float(np.nanstd(c, ddof=1) / np.sqrt(max(1, np.isfinite(c).sum())))


def compute_M_from_equal_carrier(csv_quantum: Path, *, tol: float = 1e-3) -> Tuple[float, float]:
    df = pd.read_csv(csv_quantum)
    if "R_env" not in df.columns:
        raise ValueError("CSV must contain R_env column")
    r = np.asarray(df["R_env"].values, dtype=float)
    lift = np.maximum(0.0, r - 1.0)
    # Gate by equal-carrier tolerance if available
    gate = np.ones_like(lift)
    if "rel_j_err" in df.columns:
        gate = (np.asarray(df["rel_j_err"].values, dtype=float) <= tol).astype(float)
    mvals = lift * gate
    # Normalize by max lift within the sweep (avoid cross-condition leakage)
    denom = np.nanmax(mvals) if np.nanmax(mvals) > 0 else 1.0
    mnorm = mvals / denom
    return float(np.nanmean(mnorm)), float(np.nanstd(mnorm, ddof=1) / np.sqrt(max(1, np.isfinite(mnorm).sum())))


def compute_simplex(
    *,
    classical_csv: Path,
    classical_parametric_csv: Path,
    quantum_equal_carrier_csv: Path,
    tol: float = 1e-3,
) -> SimplexResult:
    S, S_sem = compute_S_from_psd_nrmse(classical_csv)
    C, C_sem = compute_C_from_band_delta(classical_parametric_csv)
    M, M_sem = compute_M_from_equal_carrier(quantum_equal_carrier_csv, tol=tol)
    return SimplexResult(S=S, C=C, M=M, S_sem=S_sem, C_sem=C_sem, M_sem=M_sem)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Compute S/C/M simplex scores")
    p.add_argument("--classical-csv", type=Path, default=Path("results/theta_sweep_today.csv"))
    p.add_argument("--classical-parametric-csv", type=Path, default=Path("results/adaptive/classC_fraction_attempt.csv"))
    p.add_argument("--quantum-equal-carrier-csv", type=Path, default=Path("results/adaptive/kerr08_det12_theta_grid7.csv"))
    p.add_argument("--tol", type=float, default=1e-3)
    args = p.parse_args()

    res = compute_simplex(
        classical_csv=args.classical_csv,
        classical_parametric_csv=args.classical_parametric_csv,
        quantum_equal_carrier_csv=args.quantum_equal_carrier_csv,
        tol=args.tol,
    )
    b = res.barycentric()
    print("Simplex S,C,M:", res.S, res.C, res.M)
    print("SEM S,C,M:", res.S_sem, res.C_sem, res.M_sem)
    print("Barycentric:", b)


if __name__ == "__main__":
    main()

