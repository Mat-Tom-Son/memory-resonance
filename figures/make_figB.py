#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats

from analysis.stats import bootstrap_ci, cohen_d


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


def load_equal_carrier_rows(csv_path: Path) -> tuple[list[dict[str, str]], str, str, set[str]]:
    rows: list[dict[str, str]] = []
    config_hashes: set[str] = set()
    psd_norms: set[str] = set()
    solvers: set[str] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("model") != "pseudomode":
                continue
            if row.get("calibration") != "equal_carrier":
                continue
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
            solver = row.get("solver")
            if solver:
                solvers.add(solver.lower())

    if not rows:
        raise ValueError(f"No equal-carrier rows found in {csv_path}")
    if len(config_hashes) != 1:
        raise ValueError(f"Expected single config_hash but found {config_hashes}")
    if len(psd_norms) != 1:
        raise ValueError(f"Expected single psd_norm but found {psd_norms}")
    return rows, config_hashes.pop(), psd_norms.pop(), solvers


def aggregate_by_theta(rows: list[dict[str, str]]) -> tuple[dict[float, np.ndarray], dict[float, set[tuple[str, str]]]]:
    values: dict[float, list[float]] = defaultdict(list)
    counts: dict[float, set[tuple[str, str]]] = defaultdict(set)

    for row in rows:
        theta = _safe_float(row.get("theta"))
        ratio = _safe_float(row.get("baseband_ratio"))
        if theta is None or ratio is None:
            continue
        values[theta].append(ratio)
        counts[theta].add((row.get("seed"), row.get("run_id")))

    arrays = {theta: np.asarray(vals, dtype=float) for theta, vals in values.items()}
    return arrays, counts


def holm_adjust(p_values: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    prev = 1.0
    for rank, (label, pval) in enumerate(ordered, start=1):
        adj = min(1.0, (m - rank + 1) * pval)
        adj = min(adj, prev)
        adjusted[label] = adj
        prev = adj
    return adjusted


def run_peak_vs_sides_tests(data: dict[float, np.ndarray]) -> None:
    if not data:
        return
    sorted_thetas = sorted(data.keys())
    peak_theta = min(sorted_thetas, key=lambda t: abs(t - 1.0))
    peak_values = data[peak_theta]
    idx = sorted_thetas.index(peak_theta)
    lower_theta = sorted_thetas[max(idx - 1, 0)] if idx > 0 else sorted_thetas[0]
    upper_theta = sorted_thetas[min(idx + 1, len(sorted_thetas) - 1)] if idx < len(sorted_thetas) - 1 else sorted_thetas[-1]
    comparisons = []

    for side_theta in {lower_theta, upper_theta} - {peak_theta}:
        side_vals = data[side_theta]
        t_stat, p_value = stats.ttest_ind(peak_values, side_vals, equal_var=False)
        effect = cohen_d(peak_values, side_vals)
        comparisons.append((f"{peak_theta:.3f}_vs_{side_theta:.3f}", p_value, effect, side_theta))

    holm_inputs = [(label, p_val) for label, p_val, _, _ in comparisons]
    adjusted = holm_adjust(holm_inputs)

    for label, p_value, effect, side_theta in comparisons:
        print(
            f"[FIGB] peak_vs_side label={label} p={p_value:.4g} p_holm={adjusted[label]:.4g} d={effect:.3f}",
            flush=True,
        )


def build_plot(
    data: dict[float, np.ndarray],
    counts: dict[float, set[tuple[str, str]]],
    output_path: Path,
    config_hash: str,
    psd_norm: str,
    deterministic: bool,
) -> None:
    theta_vals = sorted(data.keys())
    means = []
    ci_low = []
    ci_high = []
    n_counts = []

    for theta in theta_vals:
        values = data[theta]
        means.append(np.mean(values))
        if deterministic:
            ci_low.append(means[-1])
            ci_high.append(means[-1])
        else:
            ci = bootstrap_ci(values)
            ci_low.append(ci[0])
            ci_high.append(ci[1])
        n_counts.append(len(counts.get(theta, set())))
        print(f"[FIGB] theta={theta:.3f} N_effective={n_counts[-1]}", flush=True)

    theta_arr = np.asarray(theta_vals)
    mean_arr = np.asarray(means)
    ci_low_arr = np.asarray(ci_low)
    ci_high_arr = np.asarray(ci_high)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    label = "Analytic (Gaussian solver)" if deterministic else "Baseband ratio"
    ax.plot(theta_arr, mean_arr, marker="o", color="tab:purple", label=label)
    if not deterministic:
        ax.fill_between(theta_arr, ci_low_arr, ci_high_arr, color="tab:purple", alpha=0.2)

    for theta, mean_value, n_val in zip(theta_arr, mean_arr, n_counts):
        ax.annotate(
            f"N={n_val}",
            (theta, mean_value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color="tab:purple",
        )

    counts_label = "N per Θ: " + ", ".join(f"{theta:.2f}={n}" for theta, n in zip(theta_arr, n_counts))
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_handles.append(Line2D([], [], color='none', label=counts_label))
    legend_labels.append(counts_label)
    ax.legend(legend_handles, legend_labels, loc="best")

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\Theta = \omega_1 \tau_B$")
    ax.set_ylabel("Baseband ratio")
    ax.set_title("Equal-carrier sweep")
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Figure B - Equal Carrier",
        "Author": "warm-noise-proxy pipeline",
        "Subject": f"Config hash {config_hash}",
        "Keywords": f"equal_carrier, psd_norm={psd_norm}",
    }
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, metadata=metadata)
    plt.close(fig)
    print(f"[FIGB] Figure saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure B (Equal-carrier)")
    parser.add_argument("--csv", type=Path, default=Path("results/theta_sweep_today.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures/figB_equal_carrier.pdf"))
    args = parser.parse_args()

    rows, config_hash, psd_norm, solvers = load_equal_carrier_rows(args.csv)
    if len(solvers) > 1:
        raise ValueError(f"Mixed solver types discovered for Figure B: {solvers}")
    deterministic = solvers == {"gaussian"}
    data, counts = aggregate_by_theta(rows)
    if not deterministic:
        run_peak_vs_sides_tests(data)
    else:
        print("[FIGB] Gaussian solver detected – skipping statistical tests (analytic curve).", flush=True)
    build_plot(data, counts, args.output, config_hash, psd_norm, deterministic)


if __name__ == "__main__":
    main()
