"""
Gaussian solver for linear bosonic chains used in the warm-noise proxy.

Provides utilities to build drift/diffusion matrices, compute steady-state
covariances, and evaluate power spectral densities without time-domain
trajectory simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_lyapunov


def symplectic_form(n_modes: int) -> NDArray[np.float64]:
    """Return block-diagonal symplectic form Ω for n modes."""
    blocks = []
    for _ in range(n_modes):
        blocks.append(np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float))
    return np.block([[blocks[i] if i == j else np.zeros((2, 2)) for j in range(n_modes)] for i in range(n_modes)])


@dataclass
class GaussianModel:
    drift: NDArray[np.float64]
    diffusion: NDArray[np.float64]
    x_selector: NDArray[np.float64]
    fast_selector: NDArray[np.float64]
    n_modes: int


def build_model(
    *,
    frequencies: Sequence[float],
    couplings: Iterable[tuple[int, int, float]],
    dampings: Sequence[float],
    nbar_list: Sequence[float],
    fast_mode: int,
    x_measure_mode: int,
) -> GaussianModel:
    """
    Build the Gaussian drift/diffusion matrices for a set of coupled modes.

    Parameters
    ----------
    frequencies:
        Harmonic frequencies ω_j for each mode.
    couplings:
        Iterable of (j, k, g) beam-splitter couplings g (a_j^† a_k + h.c.).
    dampings:
        Damping rates γ_j for each mode.
    nbar_list:
        Thermal occupations n̄_j associated with each damping term.
    fast_mode:
        Index of the fast mode (used for J_w1 checks).
    x_measure_mode:
        Index of the mode whose position quadrature is reported (x_3).
    """

    n_modes = len(frequencies)
    size = 2 * n_modes

    G = np.zeros((size, size), dtype=float)
    for idx, omega in enumerate(frequencies):
        x = 2 * idx
        p = x + 1
        G[x, x] = omega
        G[p, p] = omega

    for j, k, g in couplings:
        xj, pj = 2 * j, 2 * j + 1
        xk, pk = 2 * k, 2 * k + 1
        G[xj, xk] += g
        G[xk, xj] += g
        G[pj, pk] += g
        G[pk, pj] += g

    gamma_diag = np.zeros((size, size), dtype=float)
    for idx, gamma in enumerate(dampings):
        x = 2 * idx
        p = x + 1
        gamma_diag[x, x] = gamma
        gamma_diag[p, p] = gamma

    omega_mat = symplectic_form(n_modes)
    drift = omega_mat @ G - gamma_diag

    diffusion = np.zeros((size, size), dtype=float)
    for idx, (gamma, nbar) in enumerate(zip(dampings, nbar_list)):
        x = 2 * idx
        block = 2.0 * gamma * (2.0 * nbar + 1.0) * np.eye(2)
        diffusion[x : x + 2, x : x + 2] += block

    x_selector = np.zeros((1, size), dtype=float)
    x_selector[0, 2 * x_measure_mode] = 1.0

    fast_selector = np.zeros((1, size), dtype=float)
    fast_selector[0, 2 * fast_mode] = 1.0

    return GaussianModel(
        drift=drift,
        diffusion=diffusion,
        x_selector=x_selector,
        fast_selector=fast_selector,
        n_modes=n_modes,
    )


def steady_covariance(model: GaussianModel) -> NDArray[np.float64]:
    """Solve A V + V Aᵀ + D = 0 for the steady-state covariance."""
    return solve_continuous_lyapunov(model.drift, -model.diffusion)


def psd(
    model: GaussianModel,
    selector: NDArray[np.float64],
    omega: float,
) -> float:
    """
    Power spectral density for a scalar observable defined by `selector`.

    Parameters
    ----------
    selector:
        Row vector picking the observable (e.g., x₃).
    omega:
        Angular frequency at which to evaluate S(ω).
    """
    A = model.drift
    D = model.diffusion
    size = A.shape[0]
    eye = np.eye(size)
    mat = 1j * omega * eye - A
    mat_conj = -1j * omega * eye - A.T
    sol = np.linalg.solve(mat, D)
    val = selector @ sol @ np.linalg.solve(mat_conj, selector.T)
    return float(np.real(val))


def integrate_psd(
    model: GaussianModel,
    selector: NDArray[np.float64],
    omega_min: float,
    omega_max: float,
    num: int = 512,
) -> float:
    """Integrate the PSD over [ω_min, ω_max] returning the variance contribution."""
    if omega_max <= omega_min:
        return 0.0
    omegas = np.linspace(omega_min, omega_max, num, dtype=float)
    values = np.array([psd(model, selector, w) for w in omegas], dtype=float)
    integral = np.trapezoid(values, omegas)
    return float(integral / (2.0 * np.pi))


def quadrature_variance(V: NDArray[np.float64], mode: int, component: str = "x") -> float:
    """Extract variance of requested quadrature for the given mode."""
    idx = 2 * mode + (0 if component == "x" else 1)
    return float(V[idx, idx])


def mode_occupancy(V: NDArray[np.float64], mode: int) -> float:
    """Compute ⟨a† a⟩ assuming quadratures scale as x=a+a†, p=-i(a-a†)."""
    x_var = quadrature_variance(V, mode, "x")
    p_var = quadrature_variance(V, mode, "p")
    return max((x_var + p_var - 2.0) / 4.0, 0.0)
