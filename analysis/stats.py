"""
Statistical helpers for envelope analysis.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    samples: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Compute a bootstrap confidence interval for the given statistic.

    Parameters
    ----------
    samples:
        One-dimensional array of sample values.
    statistic:
        Callable that maps sampled values to a scalar statistic.
    confidence_level:
        Desired coverage (default 95%).
    n_resamples:
        Number of bootstrap resamples.
    rng:
        Optional NumPy random generator.
    """
    data = np.asarray(samples, dtype=float).ravel()
    if data.size == 0:
        raise ValueError("Cannot bootstrap an empty sample set.")
    if rng is None:
        rng = np.random.default_rng()

    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        resample = rng.choice(data, size=data.size, replace=True)
        stats[i] = float(statistic(resample))

    alpha = (1.0 - confidence_level) / 2.0
    lower = np.quantile(stats, alpha)
    upper = np.quantile(stats, 1.0 - alpha)
    return float(lower), float(upper)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two independent samples.

    Parameters
    ----------
    x, y:
        Sample arrays.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError("Samples for Cohen's d must be non-empty.")

    mean_diff = np.mean(x) - np.mean(y)
    var_pooled = (
        ((x.size - 1) * np.var(x, ddof=1) + (y.size - 1) * np.var(y, ddof=1))
        / (x.size + y.size - 2)
    )
    if var_pooled <= 0:
        return 0.0
    return float(mean_diff / np.sqrt(var_pooled))


def tost_equivalence_paired(
    x: np.ndarray,
    y: np.ndarray,
    epsilon_rel: float = 0.02,
    alpha: float = 0.05,
) -> tuple[bool, float, float]:
    """
    Two One-Sided Tests (TOST) for equivalence of paired samples.

    Tests H0: |μ_x - μ_y| ≥ ε  vs  H1: |μ_x - μ_y| < ε
    using paired differences Δ = x - y.

    Parameters
    ----------
    x, y:
        Paired sample arrays.
    epsilon_rel:
        Equivalence margin as fraction of mean(x). E.g., 0.02 = 2%.
    alpha:
        Significance level for each one-sided test (default 0.05).

    Returns
    -------
    tuple[bool, float, float]
        (passes, p_lower, p_upper)
        - passes: True if both one-sided tests reject (equivalence established)
        - p_lower: p-value for lower bound test
        - p_upper: p-value for upper bound test
    """
    from scipy import stats as sp_stats

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        return False, 1.0, 1.0
    if x.size != y.size:
        raise ValueError(f"Paired samples must have equal length (got {x.size} vs {y.size}).")

    delta = x - y
    mean_delta = np.mean(delta)
    se_delta = np.std(delta, ddof=1) / np.sqrt(delta.size)

    # Equivalence margin based on mean of x
    mean_x = np.mean(x)
    if mean_x == 0:
        epsilon = epsilon_rel  # fallback to absolute if mean is zero
    else:
        epsilon = abs(epsilon_rel * mean_x)

    if se_delta <= 0:
        # No variance - check if mean difference is within bounds
        passes = abs(mean_delta) < epsilon
        return passes, (0.0 if passes else 1.0), (0.0 if passes else 1.0)

    # TOST: two one-sided t-tests
    # H0_lower: μ_Δ ≤ -ε  vs  H1_lower: μ_Δ > -ε
    # H0_upper: μ_Δ ≥ +ε  vs  H1_upper: μ_Δ < +ε

    t_lower = (mean_delta + epsilon) / se_delta  # Test if Δ > -ε
    t_upper = (mean_delta - epsilon) / se_delta  # Test if Δ < +ε

    df = delta.size - 1

    # One-sided p-values
    p_lower = float(1.0 - sp_stats.t.cdf(t_lower, df))  # P(T > t_lower)
    p_upper = float(sp_stats.t.cdf(t_upper, df))        # P(T < t_upper)

    # Pass if BOTH one-sided tests reject at alpha level
    passes = (p_lower < alpha) and (p_upper < alpha)

    return passes, p_lower, p_upper


def paired_cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d_z effect size for paired samples.

    This is appropriate when x and y are paired measurements (e.g., same OU trace
    vs its PSD-matched surrogate). The effect size is computed as:

        d_z = mean(Δ) / sd(Δ)

    where Δ = x - y is the vector of paired differences.

    Sanity check: For paired t-test, d_z = t / √n

    Parameters
    ----------
    x, y:
        Paired sample arrays (must have same length).

    Returns
    -------
    float
        Cohen's d_z for paired design.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError("Samples for paired Cohen's d must be non-empty.")
    if x.size != y.size:
        raise ValueError(f"Paired samples must have equal length (got {x.size} vs {y.size}).")

    delta = x - y
    mean_delta = np.mean(delta)
    sd_delta = np.std(delta, ddof=1)

    if sd_delta <= 0:
        return 0.0

    d_z = float(mean_delta / sd_delta)

    # Sanity check: verify d_z = t/√n relationship
    # This catches bugs where we might be using wrong denominator
    from scipy import stats as sp_stats
    t_stat, _ = sp_stats.ttest_rel(x, y)
    expected_dz = t_stat / np.sqrt(x.size) if np.isfinite(t_stat) else 0.0
    if abs(d_z - expected_dz) > 1e-8:
        import warnings
        warnings.warn(
            f"paired_cohen_d sanity check failed: d_z={d_z:.6f} but t/√n={expected_dz:.6f}. "
            "This indicates a bug in the effect size calculation."
        )

    return d_z

