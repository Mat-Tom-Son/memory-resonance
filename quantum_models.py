"""
Shared quantum oscillator models for Markovian and pseudomode simulations.
"""

from __future__ import annotations

import numpy as np
from qutip import basis, destroy, mesolve, qeye, tensor


def run_markovian_model(
    N: int = 15,
    w1: float = 1000.0,
    w2: float = 10.0,
    w3: float = 1.0,
    g12: float = 0.05,
    g23: float = 0.05,
    gamma1: float = 0.1,
    gamma2: float = 0.05,
    gamma3: float = 0.01,
    times: np.ndarray | None = None,
    excitation_amp: float = 0.1,
):
    """
    Simulate three coupled oscillators with Markovian damping.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Simulation times, expectation of x₃, expectation of n₃.
    """
    if times is None:
        times = np.linspace(0, 2.0, 1000)

    a1 = tensor(destroy(N), qeye(N), qeye(N))
    a2 = tensor(qeye(N), destroy(N), qeye(N))
    a3 = tensor(qeye(N), qeye(N), destroy(N))

    H = w1 * a1.dag() * a1 + w2 * a2.dag() * a2 + w3 * a3.dag() * a3
    H += g12 * (a1.dag() * a2 + a1 * a2.dag())
    H += g23 * (a2.dag() * a3 + a2 * a3.dag())

    c_ops = [
        np.sqrt(2 * gamma1) * a1,
        np.sqrt(2 * gamma2) * a2,
        np.sqrt(2 * gamma3) * a3,
    ]

    psi0 = tensor(basis(N, 0), basis(N, 0), basis(N, 0))
    psi0 = (psi0 + excitation_amp * tensor(basis(N, 1), basis(N, 0), basis(N, 0))).unit()

    x3 = (a3 + a3.dag()) / np.sqrt(2)
    n3 = a3.dag() * a3

    result = mesolve(H, psi0, times, c_ops, [x3, n3])
    return times, result.expect[0], result.expect[1]


def run_pseudomode_model(
    tau_B: float = 1e-3,
    N: int = 15,
    N_pseudo: int = 10,
    w1: float = 1000.0,
    w2: float = 10.0,
    w3: float = 1.0,
    g12: float = 0.05,
    g23: float = 0.05,
    gamma1: float = 0.1,
    gamma2: float = 0.05,
    gamma3: float = 0.01,
    times: np.ndarray | None = None,
    excitation_amp: float = 0.1,
):
    """
    Simulate three oscillators with a pseudomode bath for non-Markovian dynamics.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Simulation times, expectation of x₃, expectation of n₃.
    """
    if times is None:
        times = np.linspace(0, 2.0, 1000)

    a1 = tensor(destroy(N), qeye(N), qeye(N), qeye(N_pseudo))
    a2 = tensor(qeye(N), destroy(N), qeye(N), qeye(N_pseudo))
    a3 = tensor(qeye(N), qeye(N), destroy(N), qeye(N_pseudo))
    a_pseudo = tensor(qeye(N), qeye(N), qeye(N), destroy(N_pseudo))

    w_pseudo = w1
    kappa = 1.0 / tau_B
    g_pseudo = np.sqrt(kappa * gamma1 * w1 / np.pi)

    H = w1 * a1.dag() * a1 + w2 * a2.dag() * a2 + w3 * a3.dag() * a3
    H += w_pseudo * a_pseudo.dag() * a_pseudo
    H += g12 * (a1.dag() * a2 + a1 * a2.dag())
    H += g23 * (a2.dag() * a3 + a2 * a3.dag())
    H += g_pseudo * (a1.dag() * a_pseudo + a1 * a_pseudo.dag())

    c_ops = [
        np.sqrt(2 * gamma1) * a1,
        np.sqrt(2 * gamma2) * a2,
        np.sqrt(2 * gamma3) * a3,
        np.sqrt(2 * kappa) * a_pseudo,
    ]

    psi0 = tensor(basis(N, 0), basis(N, 0), basis(N, 0), basis(N_pseudo, 0))
    psi0 = (
        psi0
        + excitation_amp
        * tensor(basis(N, 1), basis(N, 0), basis(N, 0), basis(N_pseudo, 0))
    ).unit()

    x3 = (a3 + a3.dag()) / np.sqrt(2)
    n3 = a3.dag() * a3

    result = mesolve(H, psi0, times, c_ops, [x3, n3])
    return times, result.expect[0], result.expect[1]
