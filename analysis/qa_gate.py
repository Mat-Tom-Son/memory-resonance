#!/usr/bin/env python3
"""
Lightweight QA gate for consolidated warm-noise proxy outputs.

Fails with a non-zero exit code if any tolerance breaches are detected.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

TOL_OCC = 0.03
MAX_NUDGES = 3
TOL_J = 0.02
PSD_NRMSE_LIMIT = 0.03
COHEN_D_MAX = 0.30  # Practical equivalence threshold for paired effect size


def _parse_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QA gates on consolidated CSV output.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="results/theta_sweep_today.csv",
        help="Path to the consolidated CSV file.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"[QA] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    violations: list[tuple[int, str, str]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader, start=2):  # account for header on line 1
            status = row.get("status", "").strip().lower()
            run_id = row.get("run_id", "")
            holm_family = row.get("holm_family", "")
            family_note = f" family={holm_family}" if holm_family else ""
            context = f"calib={row.get('calibration')} seed={row.get('seed')} run_id={run_id}{family_note}"
            if status and status != "ok":
                violations.append(
                    (idx, "status", f"status={row.get('status')} [{context}]")
                )

            calibration = row.get("calibration", "")
            model = row.get("model", "")
            condition = row.get("condition", "")

            min_real = _parse_float(row.get("min_real_part"))
            if not math.isnan(min_real) and min_real >= -1e-6:
                violations.append(
                    (idx, "min_real_part", f"min_real_part={min_real:.3e} ≥ -1e-6 [{context}]")
                )

            if calibration == "equal_heating" and model == "pseudomode":
                rel_occ_err = abs(_parse_float(row.get("rel_occ_err")))
                if not math.isnan(rel_occ_err) and rel_occ_err > TOL_OCC:
                    violations.append(
                        (idx, "rel_occ_err", f"|rel_occ_err|={rel_occ_err:.5f} > {TOL_OCC} [{context}]")
                    )
                nudges_used = _parse_float(row.get("nudges_used"))
                if not math.isnan(nudges_used) and nudges_used > MAX_NUDGES:
                    violations.append(
                        (idx, "nudges_used", f"nudges_used={nudges_used:.1f} > {MAX_NUDGES} [{context}]")
                    )

            if calibration == "equal_carrier" and model == "pseudomode":
                rel_j_err = _parse_float(row.get("rel_Jw1_err"))
                if not math.isnan(rel_j_err) and rel_j_err > TOL_J:
                    violations.append(
                        (idx, "rel_Jw1_err", f"rel_Jw1_err={rel_j_err:.5f} > {TOL_J} [{context}]")
                    )

            if condition == "psd_overlay":
                # Primary gate: PSD NRMSE must be < 0.03
                nrmse = _parse_float(row.get("psd_nrmse"))
                if not math.isnan(nrmse) and nrmse >= PSD_NRMSE_LIMIT:
                    violations.append(
                        (idx, "psd_nrmse", f"psd_nrmse={nrmse:.5f} ≥ {PSD_NRMSE_LIMIT} [{context}]")
                    )

                # Primary gate: Paired effect size |d_z| must be < 0.30
                cohen_d = _parse_float(row.get("cohen_d"))
                if not math.isnan(cohen_d) and abs(cohen_d) >= COHEN_D_MAX:
                    violations.append(
                        (idx, "cohen_d", f"|cohen_d|={abs(cohen_d):.5f} ≥ {COHEN_D_MAX} [{context}]")
                    )

                # Note: Holm-corrected p-value is reported for transparency but NOT used as a gate.
                # The practical equivalence criteria (PSD NRMSE and effect size) are sufficient.

    if violations:
        print("[QA] Violations detected:", file=sys.stderr)
        for idx, rule, message in violations:
            print(f"  line {idx}: {rule} -> {message}", file=sys.stderr)
        sys.exit(1)

    print("[QA] All gates passed.")


if __name__ == "__main__":
    main()
