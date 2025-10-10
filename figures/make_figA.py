#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.stats import bootstrap_ci


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


def load_classical_rows(csv_path: Path) -> tuple[list[dict[str, str]], str, str]:
    rows: list[dict[str, str]] = []
    config_hashes: set[str] = set()
    psd_norms: set[str] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("model") != "classical":
                continue
            if row.get("metric") != "baseband_power":
                continue
            norm = row.get("psd_norm")
            if norm and norm not in {"onesided", "one-sided"}:
                continue
            if norm == "one-sided":
                # Normalize legacy label to match current convention
                row["psd_norm"] = "onesided"
            status = row.get("status", "ok").lower()
            if status and status != "ok":
                continue
            rows.append(row)
            cfg = row.get("config_hash")
            if cfg:
                config_hashes.add(cfg)
            norm = row.get("psd_norm")
            if norm:
                psd_norms.add(norm)

    if not rows:
        raise ValueError(f"No classical baseband rows found in {csv_path}")
    if len(config_hashes) != 1:
        raise ValueError(f"Expected single config_hash but found {config_hashes}")
    if len(psd_norms) != 1:
        raise ValueError(f"Expected single psd_norm but found {psd_norms}")
    return rows, config_hashes.pop(), psd_norms.pop()


def compute_statistics(rows: list[dict[str, str]]) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], dict[float, set[tuple[str, str]]]]:
    data = defaultdict(lambda: defaultdict(list))
    counts: dict[float, set[tuple[str, str]]] = defaultdict(set)

    for row in rows:
        condition = row.get("condition")
        theta_val = _safe_float(row.get("theta"))
        value = _safe_float(row.get("value"))
        if condition not in {"ou", "psd_matched"} or theta_val is None or value is None:
            continue
        data[condition][theta_val].append(value)
        counts[theta_val].add((row.get("seed"), row.get("run_id")))

    stats = {
        condition: {
            theta: np.asarray(values, dtype=float)
            for theta, values in sorted(theta_map.items())
        }
        for condition, theta_map in data.items()
    }
    return stats, counts, counts


def build_plot(stats: dict[str, dict[float, np.ndarray]], counts: dict[float, set[tuple[str, str]]], output_path: Path, config_hash: str, psd_norm: str) -> None:
    thetas = sorted(stats.get("ou", {}).keys())
    if not thetas:
        raise ValueError("No OU data available for plotting.")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    legend_entries = []

    for condition, color in (("ou", "tab:blue"), ("psd_matched", "tab:orange")):
        theta_vals = []
        means = []
        ci_low = []
        ci_high = []
        n_counts = []
        for theta in thetas:
            values = stats.get(condition, {}).get(theta)
            if values is None or values.size == 0:
                continue
            theta_vals.append(theta)
            means.append(np.mean(values))
            ci = bootstrap_ci(values)
            ci_low.append(ci[0])
            ci_high.append(ci[1])
            if condition == "ou":
                n_counts.append(len(counts.get(theta, set())))
        if not theta_vals:
            continue
        theta_arr = np.asarray(theta_vals)
        mean_arr = np.asarray(means)
        ci_low_arr = np.asarray(ci_low)
        ci_high_arr = np.asarray(ci_high)
        ax.plot(theta_arr, mean_arr, marker="o", color=color, label=condition.replace("_", " ").title())
        ax.fill_between(theta_arr, ci_low_arr, ci_high_arr, color=color, alpha=0.2)
        if condition == "ou":
            for theta, mean_value, n_val in zip(theta_arr, mean_arr, n_counts):
                ax.annotate(
                    f"N={n_val}",
                    (theta, mean_value),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=9,
                    color=color,
                )
                print(f"[FIGA] theta={theta:.3f} N_effective={n_val} condition=ou", flush=True)
        else:
            for theta, mean_value in zip(theta_arr, mean_arr):
                n_val = len(counts.get(theta, set()))
                print(f"[FIGA] theta={theta:.3f} N_effective={n_val} condition=psd_matched", flush=True)

    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\Theta = \omega_1 \tau_B$")
    ax.set_ylabel("Baseband power")
    ax.set_title("Classical OU vs PSD-matched surrogate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Figure A - Classical Control",
        "Author": "warm-noise-proxy pipeline",
        "Subject": f"Config hash {config_hash}",
        "Keywords": f"classical, psd_norm={psd_norm}",
    }
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, metadata=metadata)
    plt.close(fig)
    print(f"[FIGA] Figure saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure A (Classical control)")
    parser.add_argument("--csv", type=Path, default=Path("results/theta_sweep_today.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures/figA_classical.pdf"))
    args = parser.parse_args()

    rows, config_hash, psd_norm = load_classical_rows(args.csv)
    stats, counts, _ = compute_statistics(rows)
    build_plot(stats, counts, args.output, config_hash, psd_norm)


if __name__ == "__main__":
    main()
