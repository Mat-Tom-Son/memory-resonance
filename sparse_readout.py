"""
Stage 5: Sparse readout / perceptron test using classical sweep results.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    data: Dict[str, List[Tuple[float, float]]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                theta = float(row["Theta"])
                r_env = float(row["R_env_mean"])
            except (KeyError, ValueError):
                continue
            calibration = row.get("calibration", "unknown")
            data.setdefault(calibration, []).append((theta, r_env))

    processed: Dict[str, Dict[str, np.ndarray]] = {}
    for calibration, points in data.items():
        points.sort(key=lambda item: item[0])
        theta_vals, r_env_vals = zip(*points)
        processed[calibration] = {
            "theta": np.array(theta_vals),
            "r_env": np.array(r_env_vals),
        }
    return processed


def generate_patterns(
    distribution: Dict[str, np.ndarray],
    n_samples: int,
    sparsity: float,
    threshold: float,
    rng: np.random.Generator,
) -> np.ndarray:
    theta = distribution["theta"]
    r_env = distribution["r_env"]
    n_features = len(theta)
    k = max(1, int(round(sparsity * n_features)))

    weights = np.clip(r_env - threshold, a_min=0.0, a_max=None)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    probabilities = weights / weights.sum()

    patterns = np.zeros((n_samples, n_features), dtype=float)
    for i in range(n_samples):
        active = rng.choice(n_features, size=k, replace=False, p=probabilities)
        patterns[i, active] = r_env[active]
    return patterns


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = X.shape[0]
    indices = rng.permutation(n_samples)
    test_size = max(1, int(test_fraction * n_samples))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_perceptron(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 50,
    lr: float = 0.1,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, float]:
    if rng is None:
        rng = np.random.default_rng()

    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(epochs):
        for idx in rng.permutation(n_samples):
            activation = np.dot(weights, X[idx]) + bias
            pred = 1 if activation >= 0 else 0
            update = y[idx] - pred
            if update != 0:
                weights += lr * update * X[idx]
                bias += lr * update
    return weights, bias


def evaluate_perceptron(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float) -> float:
    activation = X @ weights + bias
    preds = (activation >= 0).astype(int)
    return float(np.mean(preds == y))


def build_dataset(
    base_distribution: Dict[str, np.ndarray],
    structured_distribution: Dict[str, np.ndarray],
    sparsity: float,
    threshold: float,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    white_patterns = generate_patterns(base_distribution, n_samples, sparsity, threshold, rng)
    structured_patterns = generate_patterns(structured_distribution, n_samples, sparsity, threshold, rng)
    X = np.vstack([white_patterns, structured_patterns])
    y = np.concatenate([np.zeros(n_samples, dtype=int), np.ones(n_samples, dtype=int)])
    return X, y


def perceptron_accuracy(
    base_distribution: Dict[str, np.ndarray],
    structured_distribution: Dict[str, np.ndarray],
    sparsity: float,
    threshold: float,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    X, y = build_dataset(base_distribution, structured_distribution, sparsity, threshold, n_samples, rng)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_fraction=0.3, rng=rng)
    weights, bias = train_perceptron(X_train, y_train, epochs=60, lr=0.1, rng=rng)
    return evaluate_perceptron(X_test, y_test, weights, bias), weights, bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sparse readout perceptron benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, required=True, help="Classical summary CSV.")
    parser.add_argument(
        "--positivity_threshold",
        type=float,
        default=0.9,
        help="Threshold used to emphasise structured envelope contributions.",
    )
    parser.add_argument(
        "--sparsity_levels",
        nargs="+",
        type=float,
        default=[0.1, 0.05, 0.02],
        help="Active fraction of Θ nodes per pattern.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=300,
        help="Number of patterns per class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sparse_coding",
        help="Destination for plots and CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(Path(args.data))
    if not summary:
        raise RuntimeError(f"No calibrations found in {args.data}")

    baseline_key = None
    for key in summary:
        if "equal_variance" in key:
            baseline_key = key
            break
    if baseline_key is None:
        baseline_key = next(iter(summary))
    base_distribution = summary[baseline_key]

    calibrations = [key for key in summary if key != baseline_key]

    results: Dict[str, List[dict[str, float]]] = {cal: [] for cal in calibrations}
    best_record: tuple[float, str, np.ndarray, np.ndarray] | None = None

    for sparsity in args.sparsity_levels:
        for calibration in calibrations:
            structured_distribution = summary[calibration]

            acc_structured, weights_structured, bias_structured = perceptron_accuracy(
                base_distribution,
                structured_distribution,
                sparsity,
                args.positivity_threshold,
                args.n_samples,
                rng,
            )

            acc_white, _, _ = perceptron_accuracy(
                base_distribution,
                base_distribution,
                sparsity,
                args.positivity_threshold,
                args.n_samples,
                rng,
            )

            results[calibration].append(
                {
                    "sparsity": sparsity,
                    "accuracy_white": acc_white,
                    "accuracy_structured": acc_structured,
                }
            )

            if best_record is None or acc_structured > best_record[0]:
                best_record = (acc_structured, calibration, weights_structured, structured_distribution["theta"])

    csv_path = output_dir / "sparse_readout_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sparsity", "calibration", "accuracy_white", "accuracy_structured"])
        for calibration, rows in results.items():
            for row in rows:
                writer.writerow(
                    [
                        f"{row['sparsity']:.6g}",
                        calibration,
                        f"{row['accuracy_white']:.6g}",
                        f"{row['accuracy_structured']:.6g}",
                    ]
                )
    print(f"✓ Sparse readout CSV saved to {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sparsity_sorted = sorted(args.sparsity_levels, reverse=True)
    for calibration, rows in results.items():
        rows_sorted = sorted(rows, key=lambda item: item["sparsity"], reverse=True)
        accuracies = [row["accuracy_structured"] for row in rows_sorted]
        ax.plot(sparsity_sorted, accuracies, marker="o", linewidth=2, label=f"{calibration} (structured)")
    baseline_accuracies = [np.mean([row["accuracy_white"] for row in rows]) for rows in results.values()]
    baseline_mean = float(np.mean(baseline_accuracies)) if baseline_accuracies else 0.5
    ax.axhline(baseline_mean, color="gray", linestyle="--", linewidth=1.5, label="White-only baseline")
    ax.set_xlabel("Sparsity level (active fraction)", fontsize=12)
    ax.set_ylabel("Perceptron accuracy", fontsize=12)
    ax.set_title("Sparse readout accuracy vs sparsity", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    figure_path = output_dir / "accuracy_vs_sparsity.png"
    fig.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Accuracy plot saved to {figure_path}")

    if best_record is not None:
        acc_best, calibration_best, weights_best, theta_best = best_record
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(theta_best, weights_best, color="darkslateblue", linewidth=2)
        ax.set_xlabel("Θ = ω₁ τ_B", fontsize=11)
        ax.set_ylabel("Perceptron weight", fontsize=11)
        ax.set_title(f"Perceptron weights (calibration={calibration_best}, accuracy={acc_best:.2%})", fontsize=12)
        ax.grid(True, alpha=0.3)
        weights_path = output_dir / "perceptron_weights.png"
        fig.savefig(weights_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Perceptron weights plot saved to {weights_path}")


if __name__ == "__main__":
    main()
