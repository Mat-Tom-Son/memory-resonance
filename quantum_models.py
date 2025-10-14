"""
Shared quantum oscillator models for Markovian and pseudomode simulations.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from qutip import (
    Options,
    Qobj,
    basis,
    destroy,
    coherent,
    ket2dm,
    mesolve,
    qeye,
    tensor,
)
try:  # QuTiP 4.x
    from qutip.metrics import tracedist
except ImportError:  # QuTiP 5.x
    from qutip.core.metrics import tracedist

W1 = 1000.0
W2 = 10.0
W3 = 1.0
G12 = 0.05
G23 = 0.05
GAMMA1 = 0.1
GAMMA2 = 0.05
GAMMA3 = 0.01


def _resolve_excitation_amp(calibration_mode: str | None, base_amp: float) -> float:
    if calibration_mode is None:
        return base_amp
    if calibration_mode == "equal_carrier":
        return base_amp
    if calibration_mode == "equal_heating":
        return base_amp * 1.35
    if calibration_mode == "equal_variance":
        return base_amp * 0.9
    return base_amp


def _calibrate_pseudomode_coupling(
    tau_B: float,
    calibration_mode: str | None,
    w1: float,
    gamma1: float,
) -> float:
    kappa = 1.0 / max(tau_B, 1e-6)
    base = np.sqrt(kappa * gamma1 * w1 / np.pi)
    if calibration_mode is None:
        return base
    if calibration_mode == "equal_carrier":
        return base
    if calibration_mode == "equal_heating":
        return base * 1.2
    if calibration_mode == "equal_variance":
        return base * 0.85
    return base


def pseudomode_lorentzian_j(g: float, kappa: float, delta: float = 0.0) -> float:
    """Lorentzian spectral weight for a single pseudomode."""
    denom = delta**2 + (kappa / 2.0) ** 2
    if denom == 0:
        return float("inf")
    return float(g**2 * kappa / denom)


def calibrate_equal_carrier_g(
    kappa: float,
    baseline_kappa: float,
    baseline_g: float,
    delta: float = 0.0,
    tolerance: float = 0.02,
    max_iter: int = 6,
) -> tuple[float, float, float, float]:
    """
    Compute a coupling g for the pseudomode such that the spectral weight at ω₁
    matches that of a baseline (κ₀, g₀) pair.
    """
    if kappa <= 0 or baseline_kappa <= 0:
        raise ValueError("kappa values must be positive")
    target_j = pseudomode_lorentzian_j(baseline_g, baseline_kappa, delta)
    denom_new = delta**2 + (kappa / 2.0) ** 2
    if denom_new == 0:
        raise ValueError("Invalid parameters leading to division by zero")
    g_guess = np.sqrt(target_j * denom_new / kappa)

    g_current = float(g_guess)
    for _ in range(max_iter):
        j_est = pseudomode_lorentzian_j(g_current, kappa, delta)
        rel_err = abs(j_est - target_j) / target_j if target_j > 0 else 0.0
        if rel_err <= tolerance:
            return g_current, target_j, j_est, rel_err
        correction = np.sqrt(target_j / j_est)
        g_current *= correction

    raise RuntimeError(
        f"Failed to match spectral weight within tolerance {tolerance:.2%}: "
        f"rel_err={rel_err:.3%}"
    )


def baseline_pseudomode_coupling(
    tau_B: float,
    calibration_mode: str | None,
    w1: float,
    gamma1: float,
) -> float:
    """Expose baseline coupling used by run_pseudomode_model for reuse."""
    return _calibrate_pseudomode_coupling(tau_B, calibration_mode, w1, gamma1)


def build_markovian_operators(
    N: int = 15,
    w1: float = W1,
    w2: float = W2,
    w3: float = W3,
    g12: float = G12,
    g23: float = G23,
    gamma1: float = GAMMA1,
    gamma2: float = GAMMA2,
    gamma3: float = GAMMA3,
    nbar: float = 0.0,
) -> dict[str, Qobj]:
    """
    Construct Hamiltonian and collapse operators for the Markovian hierarchy,
    returning the quadrature operators required for spectral analysis.
    """
    a1 = tensor(destroy(N), qeye(N), qeye(N))
    a2 = tensor(qeye(N), destroy(N), qeye(N))
    a3 = tensor(qeye(N), qeye(N), destroy(N))

    H = w1 * a1.dag() * a1 + w2 * a2.dag() * a2 + w3 * a3.dag() * a3
    H += g12 * (a1.dag() * a2 + a1 * a2.dag())
    H += g23 * (a2.dag() * a3 + a2 * a3.dag())

    if nbar > 0.0:
        c_ops = [
            np.sqrt(2 * gamma1 * (nbar + 1.0)) * a1,
            np.sqrt(2 * gamma1 * nbar) * a1.dag(),
            np.sqrt(2 * gamma2) * a2,
            np.sqrt(2 * gamma3) * a3,
        ]
    else:
        c_ops = [
            np.sqrt(2 * gamma1) * a1,
            np.sqrt(2 * gamma2) * a2,
            np.sqrt(2 * gamma3) * a3,
        ]

    x3 = (a3 + a3.dag()) / np.sqrt(2.0)
    x_fast = (a1 + a1.dag()) / np.sqrt(2.0)
    n1_op = a1.dag() * a1
    n3_op = a3.dag() * a3

    return {
        "H": H,
        "c_ops": c_ops,
        "x3": x3,
        "x_fast": x_fast,
        "n1_op": n1_op,
        "n3_op": n3_op,
        "a1": a1,
        "a3": a3,
    }


def build_pseudomode_operators(
    tau_B: float,
    N: int = 15,
    N_pseudo: int = 10,
    w1: float = W1,
    w2: float = W2,
    w3: float = W3,
    g12: float = G12,
    g23: float = G23,
    gamma1: float = GAMMA1,
    gamma2: float = GAMMA2,
    gamma3: float = GAMMA3,
    calibration_mode: str | None = None,
    gpm_scale: float = 1.0,
    g_override: float | None = None,
    nbar: float = 0.0,
    omega_c: float | None = None,
    nbar_pseudo: float | None = None,
    kerr_fast: float = 0.0,
) -> dict[str, Qobj]:
    """
    Construct Hamiltonian/collapse operators for the pseudomode hierarchy.
    """
    if tau_B <= 0:
        raise ValueError("tau_B must be positive for pseudomode construction")

    a1 = tensor(destroy(N), qeye(N), qeye(N), qeye(N_pseudo))
    a2 = tensor(qeye(N), destroy(N), qeye(N), qeye(N_pseudo))
    a3 = tensor(qeye(N), qeye(N), destroy(N), qeye(N_pseudo))
    a_pseudo = tensor(qeye(N), qeye(N), qeye(N), destroy(N_pseudo))

    omega_c_value = float(omega_c) if omega_c is not None else w1
    kappa = 1.0 / tau_B
    if g_override is not None:
        g_pseudo = float(g_override)
    else:
        g_base = _calibrate_pseudomode_coupling(tau_B, calibration_mode, w1, gamma1)
        g_pseudo = g_base * float(gpm_scale)

    H = w1 * a1.dag() * a1 + w2 * a2.dag() * a2 + w3 * a3.dag() * a3
    H += omega_c_value * a_pseudo.dag() * a_pseudo
    H += g12 * (a1.dag() * a2 + a1 * a2.dag())
    H += g23 * (a2.dag() * a3 + a2 * a3.dag())
    H += g_pseudo * (a1.dag() * a_pseudo + a1 * a_pseudo.dag())
    if kerr_fast:
        kerr = float(kerr_fast)
        H += 0.5 * kerr * a1.dag() * a1.dag() * a1 * a1

    c_ops: list[Qobj] = []
    if nbar > 0.0:
        c_ops.extend(
            [
                np.sqrt(2 * gamma1 * (nbar + 1.0)) * a1,
                np.sqrt(2 * gamma1 * nbar) * a1.dag(),
            ]
        )
    else:
        c_ops.append(np.sqrt(2 * gamma1) * a1)

    c_ops.extend(
        [
            np.sqrt(2 * gamma2) * a2,
            np.sqrt(2 * gamma3) * a3,
        ]
    )

    nbar_pm = nbar if nbar_pseudo is None else float(nbar_pseudo)
    if nbar_pm > 0.0:
        c_ops.extend(
            [
                np.sqrt(2 * kappa * (nbar_pm + 1.0)) * a_pseudo,
                np.sqrt(2 * kappa * nbar_pm) * a_pseudo.dag(),
            ]
        )
    else:
        c_ops.append(np.sqrt(2 * kappa) * a_pseudo)

    x3 = (a3 + a3.dag()) / np.sqrt(2.0)
    x_fast = (a1 + a1.dag()) / np.sqrt(2.0)
    n1_op = a1.dag() * a1
    n3_op = a3.dag() * a3

    return {
        "H": H,
        "c_ops": c_ops,
        "x3": x3,
        "x_fast": x_fast,
        "n1_op": n1_op,
        "n3_op": n3_op,
        "a1": a1,
        "a3": a3,
        "a_pseudo": a_pseudo,
        "g_used": g_pseudo,
        "kappa": kappa,
        "omega_c": omega_c_value,
    }


def _build_solver_options(
    user_options: dict | None = None,
    *,
    progress_bar: str | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    nsteps: int | None = None,
    rhs_reuse: bool | None = True,
    store_states: bool | None = False,
) -> Options:
    opts = Options()

    def _set_option(name: str, value) -> None:
        if value is None:
            return
        if hasattr(opts, name):
            setattr(opts, name, value)

    _set_option("progress_bar", progress_bar)
    _set_option("store_states", store_states)
    _set_option("atol", atol)
    _set_option("rtol", rtol)
    if nsteps is not None:
        _set_option("nsteps", int(nsteps))
    if rhs_reuse is not None:
        _set_option("rhs_reuse", bool(rhs_reuse))

    if user_options:
        for key, value in user_options.items():
            _set_option(key, value)

    return opts


def run_markovian_model(
    N: int = 15,
    w1: float = W1,
    w2: float = W2,
    w3: float = W3,
    g12: float = G12,
    g23: float = G23,
    gamma1: float = GAMMA1,
    gamma2: float = GAMMA2,
    gamma3: float = GAMMA3,
    times: np.ndarray | None = None,
    excitation_amp: float = 0.1,
    calibration_mode: str | None = None,
    solver_options: dict | None = None,
    progress_bar: str | None = None,
    atol: float | None = 1e-6,
    rtol: float | None = 1e-6,
    nsteps: int | None = 50_000,
    rhs_reuse: bool | None = True,
    return_n1: bool = False,
    nbar: float = 0.0,
    a1_init_alpha: float = 0.0,
    with_stats: bool = False,
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

    excitation_amp = _resolve_excitation_amp(calibration_mode, excitation_amp)

    a1 = tensor(destroy(N), qeye(N), qeye(N))
    a2 = tensor(qeye(N), destroy(N), qeye(N))
    a3 = tensor(qeye(N), qeye(N), destroy(N))

    H = w1 * a1.dag() * a1 + w2 * a2.dag() * a2 + w3 * a3.dag() * a3
    H += g12 * (a1.dag() * a2 + a1 * a2.dag())
    H += g23 * (a2.dag() * a3 + a2 * a3.dag())

    if nbar > 0.0:
        c_ops = [
            np.sqrt(2 * gamma1 * (nbar + 1.0)) * a1,
            np.sqrt(2 * gamma1 * nbar) * a1.dag(),
            np.sqrt(2 * gamma2) * a2,
            np.sqrt(2 * gamma3) * a3,
        ]
    else:
        c_ops = [
            np.sqrt(2 * gamma1) * a1,
            np.sqrt(2 * gamma2) * a2,
            np.sqrt(2 * gamma3) * a3,
        ]

    osc1 = coherent(N, a1_init_alpha) if abs(a1_init_alpha) > 1e-12 else basis(N, 0)
    psi0 = tensor(osc1, basis(N, 0), basis(N, 0))
    if abs(excitation_amp) > 0:
        psi0 = (psi0 + excitation_amp * tensor(basis(N, 1), basis(N, 0), basis(N, 0))).unit()

    x3 = (a3 + a3.dag()) / np.sqrt(2)
    n3 = a3.dag() * a3
    n1 = a1.dag() * a1 if return_n1 else None

    e_ops = [x3, n3]
    if n1 is not None:
        e_ops.append(n1)

    options = _build_solver_options(
        solver_options,
        progress_bar=progress_bar,
        atol=atol,
        rtol=rtol,
        nsteps=nsteps,
        rhs_reuse=rhs_reuse,
        store_states=False,
    )
    result = mesolve(H, psi0, times, c_ops, e_ops, options=options)
    stats = getattr(result, "stats", {})
    x3_expect = result.expect[0]
    n3_expect = result.expect[1]
    if n1 is not None:
        n1_expect = result.expect[2]
        if with_stats:
            return times, x3_expect, n3_expect, n1_expect, stats
        return times, x3_expect, n3_expect, n1_expect
    if with_stats:
        return times, x3_expect, n3_expect, stats
    return times, x3_expect, n3_expect


def run_pseudomode_model(
    tau_B: float = 1e-3,
    N: int = 15,
    N_pseudo: int = 10,
    w1: float = W1,
    w2: float = W2,
    w3: float = W3,
    g12: float = G12,
    g23: float = G23,
    gamma1: float = GAMMA1,
    gamma2: float = GAMMA2,
    gamma3: float = GAMMA3,
    times: np.ndarray | None = None,
    excitation_amp: float = 0.1,
    calibration_mode: str | None = None,
    return_states: bool = False,
    solver_options: dict | None = None,
    progress_bar: str | None = None,
    atol: float | None = 1e-6,
    rtol: float | None = 1e-6,
    nsteps: int | None = 50_000,
    rhs_reuse: bool | None = True,
    gpm_scale: float = 1.0,
    return_n1: bool = False,
    nbar: float = 0.0,
    a1_init_alpha: float = 0.0,
    omega_c: float | None = None,
    nbar_pseudo: float | None = None,
    g_override: float | None = None,
    with_stats: bool = False,
    kerr_fast: float = 0.0,
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

    excitation_amp = _resolve_excitation_amp(calibration_mode, excitation_amp)

    a1 = tensor(destroy(N), qeye(N), qeye(N), qeye(N_pseudo))
    a2 = tensor(qeye(N), destroy(N), qeye(N), qeye(N_pseudo))
    a3 = tensor(qeye(N), qeye(N), destroy(N), qeye(N_pseudo))
    a_pseudo = tensor(qeye(N), qeye(N), qeye(N), destroy(N_pseudo))

    w_pseudo = omega_c if omega_c is not None else w1
    kappa = 1.0 / tau_B
    if g_override is not None:
        g_pseudo = float(g_override)
    else:
        g_pseudo = _calibrate_pseudomode_coupling(tau_B, calibration_mode, w1, gamma1) * gpm_scale

    H = w1 * a1.dag() * a1 + w2 * a2.dag() * a2 + w3 * a3.dag() * a3
    H += w_pseudo * a_pseudo.dag() * a_pseudo
    H += g12 * (a1.dag() * a2 + a1 * a2.dag())
    H += g23 * (a2.dag() * a3 + a2 * a3.dag())
    H += g_pseudo * (a1.dag() * a_pseudo + a1 * a_pseudo.dag())
    if kerr_fast:
        kerr = float(kerr_fast)
        H += 0.5 * kerr * a1.dag() * a1.dag() * a1 * a1

    c_ops: list[Qobj] = []
    if nbar > 0.0:
        c_ops.extend(
            [
                np.sqrt(2 * gamma1 * (nbar + 1.0)) * a1,
                np.sqrt(2 * gamma1 * nbar) * a1.dag(),
            ]
        )
    else:
        c_ops.append(np.sqrt(2 * gamma1) * a1)

    c_ops.extend(
        [
            np.sqrt(2 * gamma2) * a2,
            np.sqrt(2 * gamma3) * a3,
        ]
    )

    nbar_pm = nbar if nbar_pseudo is None else nbar_pseudo
    if nbar_pm > 0.0:
        c_ops.extend(
            [
                np.sqrt(2 * kappa * (nbar_pm + 1.0)) * a_pseudo,
                np.sqrt(2 * kappa * nbar_pm) * a_pseudo.dag(),
            ]
        )
    else:
        c_ops.append(np.sqrt(2 * kappa) * a_pseudo)

    osc1 = coherent(N, a1_init_alpha) if abs(a1_init_alpha) > 1e-12 else basis(N, 0)
    psi0 = tensor(osc1, basis(N, 0), basis(N, 0), basis(N_pseudo, 0))
    psi0 = (
        psi0
        + excitation_amp
        * tensor(basis(N, 1), basis(N, 0), basis(N, 0), basis(N_pseudo, 0))
    ).unit()

    x3 = (a3 + a3.dag()) / np.sqrt(2)
    n3 = a3.dag() * a3
    n1 = a1.dag() * a1 if return_n1 else None

    e_ops = [x3, n3]
    if n1 is not None:
        e_ops.append(n1)

    options = _build_solver_options(
        solver_options,
        progress_bar=progress_bar,
        atol=atol,
        rtol=rtol,
        nsteps=nsteps,
        rhs_reuse=rhs_reuse,
        store_states=return_states,
    )
    result = mesolve(H, psi0, times, c_ops, e_ops, options=options)
    stats = getattr(result, "stats", {})
    x3_expect = result.expect[0]
    n3_expect = result.expect[1]
    if return_states:
        if n1 is not None:
            n1_expect = result.expect[2]
            if with_stats:
                return times, x3_expect, n3_expect, n1_expect, result.states, stats
            return times, x3_expect, n3_expect, n1_expect, result.states
        if with_stats:
            return times, x3_expect, n3_expect, result.states, stats
        return times, x3_expect, n3_expect, result.states
    if n1 is not None:
        n1_expect = result.expect[2]
        if with_stats:
            return times, x3_expect, n3_expect, n1_expect, stats
        return times, x3_expect, n3_expect, n1_expect
    if with_stats:
        return times, x3_expect, n3_expect, stats
    return times, x3_expect, n3_expect


def compute_qrt_psd_markovian(
    N: int = 15,
    w1: float = W1,
    w2: float = W2,
    w3: float = W3,
    g12: float = G12,
    g23: float = G23,
    gamma1: float = GAMMA1,
    gamma2: float = GAMMA2,
    gamma3: float = GAMMA3,
    nbar: float = 0.0,
    freq_hz_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the PSD of x3 fluctuations using the Quantum Regression Theorem (QRT).

    This function calculates S_x3(ω) = ∫ <δx3(t)δx3(0)> e^{-iωt} dt
    where δx3 = x3 - <x3>_ss measures fluctuations around steady state.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (freq_hz, psd_omega, psd_one_sided)
        - freq_hz: frequency grid in Hz
        - psd_omega: two-sided PSD in rad/s units
        - psd_one_sided: one-sided PSD in Hz units
    """
    from qutip import steadystate, spectrum

    # Build operators
    a1 = tensor(destroy(N), qeye(N), qeye(N))
    a2 = tensor(qeye(N), destroy(N), qeye(N))
    a3 = tensor(qeye(N), qeye(N), destroy(N))

    H = w1 * a1.dag() * a1 + w2 * a2.dag() * a2 + w3 * a3.dag() * a3
    H += g12 * (a1.dag() * a2 + a1 * a2.dag())
    H += g23 * (a2.dag() * a3 + a2 * a3.dag())

    if nbar > 0.0:
        c_ops = [
            np.sqrt(2 * gamma1 * (nbar + 1.0)) * a1,
            np.sqrt(2 * gamma1 * nbar) * a1.dag(),
            np.sqrt(2 * gamma2) * a2,
            np.sqrt(2 * gamma3) * a3,
        ]
    else:
        c_ops = [
            np.sqrt(2 * gamma1) * a1,
            np.sqrt(2 * gamma2) * a2,
            np.sqrt(2 * gamma3) * a3,
        ]

    # Define frequency grid
    if freq_hz_array is None:
        f3_hz = w3 / (2.0 * np.pi)
        freq_hz_array = np.linspace(0.0, f3_hz * 2.5, 2000, dtype=float)

    omega_array = 2.0 * np.pi * freq_hz_array

    # Compute PSD using QRT: spectrum computes S(ω) for the x3 operator
    x3 = (a3 + a3.dag()) / np.sqrt(2.0)
    psd_omega = spectrum(H, omega_array, c_ops, x3, x3, solver='es')

    # Convert to one-sided PSD in Hz units
    psd_one_sided = np.array(psd_omega, dtype=float)
    positive_freq = freq_hz_array > 0.0
    psd_one_sided[positive_freq] *= 2.0

    return freq_hz_array, np.array(psd_omega, dtype=float), psd_one_sided


def compute_qrt_psd_pseudomode(
    tau_B: float,
    N: int = 15,
    N_pseudo: int = 10,
    w1: float = W1,
    w2: float = W2,
    w3: float = W3,
    g12: float = G12,
    g23: float = G23,
    gamma1: float = GAMMA1,
    gamma2: float = GAMMA2,
    gamma3: float = GAMMA3,
    calibration_mode: str | None = None,
    gpm_scale: float = 1.0,
    g_override: float | None = None,
    nbar: float = 0.0,
    omega_c: float | None = None,
    nbar_pseudo: float | None = None,
    freq_hz_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the PSD of x3 fluctuations for pseudomode model using QRT.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (freq_hz, psd_omega, psd_one_sided)
    """
    from qutip import steadystate, spectrum

    if tau_B <= 0:
        raise ValueError("tau_B must be positive")

    # Build operators
    a1 = tensor(destroy(N), qeye(N), qeye(N), qeye(N_pseudo))
    a2 = tensor(qeye(N), destroy(N), qeye(N), qeye(N_pseudo))
    a3 = tensor(qeye(N), qeye(N), destroy(N), qeye(N_pseudo))
    a_pseudo = tensor(qeye(N), qeye(N), qeye(N), destroy(N_pseudo))

    w_pseudo = omega_c if omega_c is not None else w1
    kappa = 1.0 / tau_B
    if g_override is not None:
        g_pseudo = float(g_override)
    else:
        g_pseudo = _calibrate_pseudomode_coupling(tau_B, calibration_mode, w1, gamma1) * gpm_scale

    H = w1 * a1.dag() * a1 + w2 * a2.dag() * a2 + w3 * a3.dag() * a3
    H += w_pseudo * a_pseudo.dag() * a_pseudo
    H += g12 * (a1.dag() * a2 + a1 * a2.dag())
    H += g23 * (a2.dag() * a3 + a2 * a3.dag())
    H += g_pseudo * (a1.dag() * a_pseudo + a1 * a_pseudo.dag())

    c_ops: list[Qobj] = []
    if nbar > 0.0:
        c_ops.extend([
            np.sqrt(2 * gamma1 * (nbar + 1.0)) * a1,
            np.sqrt(2 * gamma1 * nbar) * a1.dag(),
        ])
    else:
        c_ops.append(np.sqrt(2 * gamma1) * a1)

    c_ops.extend([
        np.sqrt(2 * gamma2) * a2,
        np.sqrt(2 * gamma3) * a3,
    ])

    nbar_pm = nbar if nbar_pseudo is None else nbar_pseudo
    if nbar_pm > 0.0:
        c_ops.extend([
            np.sqrt(2 * kappa * (nbar_pm + 1.0)) * a_pseudo,
            np.sqrt(2 * kappa * nbar_pm) * a_pseudo.dag(),
        ])
    else:
        c_ops.append(np.sqrt(2 * kappa) * a_pseudo)

    # Define frequency grid
    if freq_hz_array is None:
        f3_hz = w3 / (2.0 * np.pi)
        freq_hz_array = np.linspace(0.0, f3_hz * 2.5, 2000, dtype=float)

    omega_array = 2.0 * np.pi * freq_hz_array

    # Compute PSD using QRT
    x3 = (a3 + a3.dag()) / np.sqrt(2.0)
    psd_omega = spectrum(H, omega_array, c_ops, x3, x3, solver='es')

    # Convert to one-sided PSD in Hz units
    psd_one_sided = np.array(psd_omega, dtype=float)
    positive_freq = freq_hz_array > 0.0
    psd_one_sided[positive_freq] *= 2.0

    return freq_hz_array, np.array(psd_omega, dtype=float), psd_one_sided


def compute_BLP_non_markovianity(
    params: dict | None = None,
    states: Sequence[Qobj] | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Breuer-Laine-Piilo (BLP) non-Markovianity measure for the pseudomode model.

    Parameters
    ----------
    params : dict, optional
        Simulation parameters. Recognised keys: 'tau_B', 'N', 'N_pseudo', 'times', 'w1', 'w2',
        'w3', 'g12', 'g23', 'gamma1', 'gamma2', 'gamma3', 'calibration_mode'.
    states : sequence of Qobj, optional
        Two initial states (kets or density matrices). If omitted, a default pair of nearby
        coherent-like states on the first oscillator is used.

    Returns
    -------
    tuple
        (N_BLP, times, trace_distances)
    """
    params = params or {}
    tau_B = float(params.get("tau_B", 1e-3))
    N = int(params.get("N", 10))
    N_pseudo = int(params.get("N_pseudo", 8))
    w1 = float(params.get("w1", W1))
    w2 = float(params.get("w2", W2))
    w3 = float(params.get("w3", W3))
    g12 = float(params.get("g12", G12))
    g23 = float(params.get("g23", G23))
    gamma1 = float(params.get("gamma1", GAMMA1))
    gamma2 = float(params.get("gamma2", GAMMA2))
    gamma3 = float(params.get("gamma3", GAMMA3))
    calibration_mode = params.get("calibration_mode")

    times = params.get("times")
    if times is None:
        times = np.linspace(0, 2.0, 600)
    else:
        times = np.asarray(times, dtype=float)

    a1 = tensor(destroy(N), qeye(N), qeye(N), qeye(N_pseudo))
    a2 = tensor(qeye(N), destroy(N), qeye(N), qeye(N_pseudo))
    a3 = tensor(qeye(N), qeye(N), destroy(N), qeye(N_pseudo))
    a_pseudo = tensor(qeye(N), qeye(N), qeye(N), destroy(N_pseudo))

    gpm_scale = float(params.get("gpm_scale", 1.0))
    g_pseudo = _calibrate_pseudomode_coupling(tau_B, calibration_mode, w1, gamma1) * gpm_scale
    kappa = 1.0 / tau_B

    H = w1 * a1.dag() * a1 + w2 * a2.dag() * a2 + w3 * a3.dag() * a3
    H += w1 * a_pseudo.dag() * a_pseudo
    H += g12 * (a1.dag() * a2 + a1 * a2.dag())
    H += g23 * (a2.dag() * a3 + a2 * a3.dag())
    H += g_pseudo * (a1.dag() * a_pseudo + a1 * a_pseudo.dag())

    c_ops = [
        np.sqrt(2 * gamma1) * a1,
        np.sqrt(2 * gamma2) * a2,
        np.sqrt(2 * gamma3) * a3,
        np.sqrt(2 * kappa) * a_pseudo,
    ]

    if states is None:
        base = tensor(basis(N, 0), basis(N, 0), basis(N, 0), basis(N_pseudo, 0))
        delta = params.get("state_delta", 0.08)
        perturbed = (
            base
            + delta * tensor(basis(N, 1), basis(N, 0), basis(N, 0), basis(N_pseudo, 0))
        ).unit()
        states = (ket2dm(base), ket2dm(perturbed))
    else:
        if len(states) != 2:
            raise ValueError("states must contain exactly two initial conditions.")
        prepared_states = []
        for state in states:
            if not isinstance(state, Qobj):
                state = Qobj(state)
            if state.isket:
                prepared_states.append(ket2dm(state))
            elif state.isoper:
                prepared_states.append(state)
            else:
                raise ValueError("Each state must be a ket or density operator.")
        states = tuple(prepared_states)

    blp_solver_options = params.get("solver_options") if params else None
    blp_progress = params.get("progress_bar") if params else None
    blp_atol = params.get("atol", 1e-6) if params else 1e-6
    blp_rtol = params.get("rtol", 1e-6) if params else 1e-6
    blp_nsteps = params.get("nsteps", 50_000) if params else 50_000
    blp_rhs_reuse = params.get("rhs_reuse", True) if params else True

    options = _build_solver_options(
        blp_solver_options,
        progress_bar=blp_progress,
        atol=blp_atol,
        rtol=blp_rtol,
        nsteps=blp_nsteps,
        rhs_reuse=blp_rhs_reuse,
        store_states=True,
    )
    evolutions = []
    for rho0 in states:
        result = mesolve(H, rho0, times, c_ops, [], options=options)
        evolutions.append(result.states)

    distances = np.zeros(len(times))
    for idx, (rho_a, rho_b) in enumerate(zip(evolutions[0], evolutions[1])):
        distances[idx] = tracedist(rho_a, rho_b)

    increments = np.diff(distances)
    positive = increments[increments > 0]
    n_blp = float(np.sum(positive))
    return n_blp, times, distances
