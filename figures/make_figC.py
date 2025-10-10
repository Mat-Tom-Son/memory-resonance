#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    if np.isnan(result):
        return None
    return result


def load_quantum_rows(csv_path: Path) -> tuple[list[dict[str, str]], str, str, dict[str, float]]:
    pseud_rows: list[dict[str, str]] = []
    baseline_map: dict[str, float] = {}
    config_hashes: set[str] = set()
    psd_norms: set[str] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            model = row.get("model")
            run_id = row.get("run_id")
            cfg = row.get("config_hash")
            norm = row.get("psd_norm")
            if norm == "one-sided":
                row["psd_norm"] = "onesided"
                norm = "onesided"
            if norm and norm not in {"onesided", "twosided"}:
                continue
            if cfg:
                config_hashes.add(cfg)
            if norm:
                psd_norms.add(norm)
            if model == "lindblad":
                if run_id:
                    baseline_map[run_id] = _safe_float(row.get("narrowband_power")) or 0.0
                continue
            if model != "pseudomode":
                continue
            status = row.get("status", "ok").lower()
            if status and status != "ok":
                continue
            pseud_rows.append(row)

    if not pseud_rows:
        raise ValueError(f"No pseudomode rows found in {csv_path}")
    if len(config_hashes) != 1:
        raise ValueError(f"Expected single config_hash but found {config_hashes}")
    if len(psd_norms) != 1:
        raise ValueError(f"Expected single psd_norm but found {psd_norms}")
    return pseud_rows, config_hashes.pop(), psd_norms.pop(), baseline_map


def prepare_points(rows: list[dict[str, str]], baseline_map: dict[str, float]):
    results = defaultdict(list)
    for row in rows:
        run_id = row.get("run_id")
        baseline = baseline_map.get(run_id)
        if baseline is None or baseline <= 0:
            continue
        baseband_ratio = _safe_float(row.get("baseband_ratio"))
        nb_power = _safe_float(row.get("narrowband_power"))
        if baseband_ratio is None or nb_power is None:
            continue
        narrowband_ratio = nb_power / baseline
        calibration = row.get("calibration", "unknown")
        results[calibration].append((baseband_ratio, narrowband_ratio))
    return results


def build_plot(data: dict[str, list[tuple[float, float]]], output_path: Path, config_hash: str, psd_norm: str) -> None:
    if not data:
        raise ValueError("No data available for Figure C.")

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    for color, (calibration, points) in zip(colors, data.items()):
        arr = np.asarray(points, dtype=float)
        xvals = arr[:, 0]
        yvals = arr[:, 1]
        ax.scatter(xvals, yvals, alpha=0.65, label=f"{calibration} (N={len(points)})", color=color)
        if len(points) >= 2:
            corr = np.corrcoef(xvals, yvals)[0, 1]
        else:
            corr = float("nan")
        print(f"[FIGC] calibration={calibration} N={len(points)} corr={corr:.3f}", flush=True)

    lims = [
        0.9 * min(ax.get_xlim()[0], ax.get_ylim()[0], 0.0),
        1.1 * max(ax.get_xlim()[1], ax.get_ylim()[1], 1.2),
    ]
    ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1.0, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Baseband ratio")
    ax.set_ylabel("Narrowband ratio")
    ax.set_title("Robustness of spectral metrics")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Figure C - Robustness",
        "Author": "warm-noise-proxy pipeline",
        "Subject": f"Config hash {config_hash}",
        "Keywords": f"robustness, psd_norm={psd_norm}",
    }
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, metadata=metadata)
    plt.close(fig)
    print(f"[FIGC] Figure saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure C (Robustness)")
    parser.add_argument("--csv", type=Path, default=Path("results/theta_sweep_today.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures/figC_robustness.pdf"))
    args = parser.parse_args()

    rows, config_hash, psd_norm, baseline_map = load_quantum_rows(args.csv)
    data = prepare_points(rows, baseline_map)
    build_plot(data, args.output, config_hash, psd_norm)


if __name__ == "__main__":
    main()
