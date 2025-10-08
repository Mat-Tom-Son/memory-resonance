"""
Test suite for quantum hierarchy simulation.
Run this before long simulations to validate the environment.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from qutip import (
    __version__ as qutip_version,
    basis,
    destroy,
    expect,
    mesolve,
    qeye,
    tensor,
)
from scipy import signal

from quantum_models import run_markovian_model, run_pseudomode_model


class TestResults:
    def __init__(self) -> None:
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures: list[tuple[str, str]] = []

    def record_pass(self, test_name: str) -> None:
        self.tests_run += 1
        self.tests_passed += 1
        print(f"  ✓ {test_name}")

    def record_fail(self, test_name: str, error: str) -> None:
        self.tests_run += 1
        self.tests_failed += 1
        self.failures.append((test_name, error))
        print(f"  ✗ {test_name}: {error}")

    def summary(self) -> bool:
        print("\n" + "=" * 60)
        print(f"Tests run:   {self.tests_run}")
        print(f"Passed:      {self.tests_passed}")
        print(f"Failed:      {self.tests_failed}")
        if self.tests_failed > 0:
            print("\nFailed tests:")
            for name, error in self.failures:
                print(f"  - {name}: {error}")
            print("\n⚠️  Some tests failed. Fix these before running simulations.")
            return False

        print("\n✓ All tests passed! Ready to run simulations.")
        return True


results = TestResults()

print("=" * 60)
print("QUANTUM HIERARCHY TEST SUITE")
print("=" * 60)

# ============================================================================
# TEST 1: QuTiP Installation
# ============================================================================
print("\n[TEST 1] QuTiP Installation")
try:
    print(f"  QuTiP version: {qutip_version}")
    major_version = int(qutip_version.split(".")[0])
    if major_version < 4:
        results.record_fail("QuTiP version", f"Version {qutip_version} too old, need >= 4.0")
    else:
        results.record_pass("QuTiP installation")
except Exception as exc:  # pragma: no cover - defensive
    results.record_fail("QuTiP installation", str(exc))

# ============================================================================
# TEST 2: Basic Operator Construction
# ============================================================================
print("\n[TEST 2] Basic Operator Construction")
try:
    N = 5
    a = destroy(N)

    if a.shape != (N, N):
        results.record_fail("Operator dimensions", f"Expected ({N},{N}), got {a.shape}")
    else:
        results.record_pass("Single oscillator creation")

    state_1 = basis(N, 1)
    result_vec = a * state_1
    expected_vec = np.sqrt(1) * basis(N, 0)
    if not np.allclose(result_vec.full(), expected_vec.full()):
        results.record_fail("Ladder operator action", "a|1> != sqrt(1)|0>")
    else:
        results.record_pass("Ladder operator action")
except Exception as exc:  # pragma: no cover - defensive
    results.record_fail("Basic operators", str(exc))

# ============================================================================
# TEST 3: Tensor Product Construction
# ============================================================================
print("\n[TEST 3] Tensor Product Construction")
try:
    N = 5
    a1 = tensor(destroy(N), qeye(N), qeye(N))
    a2 = tensor(qeye(N), destroy(N), qeye(N))
    a3 = tensor(qeye(N), qeye(N), destroy(N))

    expected_dim = N**3
    if a1.shape != (expected_dim, expected_dim):
        results.record_fail(
            "Tensor product dimensions",
            f"Expected ({expected_dim},{expected_dim}), got {a1.shape}",
        )
    else:
        results.record_pass("Three-oscillator tensor product")

    commutator = a1 * a2 - a2 * a1
    if commutator.norm() > 1e-10:
        results.record_fail("Operator commutativity", f"[a1,a2] norm = {commutator.norm()}")
    else:
        results.record_pass("Operator commutativity")

    x1 = (a1 + a1.dag()) / np.sqrt(2)
    if not x1.isherm:
        results.record_fail("Position operator Hermiticity", "x1 is not Hermitian")
    else:
        results.record_pass("Position operator Hermiticity")
except Exception as exc:  # pragma: no cover - defensive
    results.record_fail("Tensor products", str(exc))

# ============================================================================
# TEST 4: Hamiltonian Construction
# ============================================================================
print("\n[TEST 4] Hamiltonian Construction")
try:
    N = 5
    a1 = tensor(destroy(N), qeye(N), qeye(N))
    a2 = tensor(qeye(N), destroy(N), qeye(N))
    a3 = tensor(qeye(N), qeye(N), destroy(N))

    w1, w2, w3 = 1000.0, 10.0, 1.0
    g12, g23 = 0.05, 0.05

    H = w1 * a1.dag() * a1 + w2 * a2.dag() * a2 + w3 * a3.dag() * a3
    H += g12 * (a1.dag() * a2 + a1 * a2.dag())
    H += g23 * (a2.dag() * a3 + a2 * a3.dag())

    if not H.isherm:
        results.record_fail("Hamiltonian Hermiticity", "H is not Hermitian")
    else:
        results.record_pass("Hamiltonian Hermiticity")

    psi0 = tensor(basis(N, 0), basis(N, 0), basis(N, 0))
    E0 = expect(H, psi0)
    if not np.isclose(E0, 0.0, atol=1e-10):
        results.record_fail("Ground state energy", f"E0 = {E0}, expected 0")
    else:
        results.record_pass("Ground state energy")

    psi1 = tensor(basis(N, 1), basis(N, 0), basis(N, 0))
    E1 = expect(H, psi1)
    if not np.isclose(E1, w1, rtol=1e-3):
        results.record_fail("First excited state", f"E1 = {E1}, expected ~{w1}")
    else:
        results.record_pass("First excited state energy")
except Exception as exc:  # pragma: no cover - defensive
    results.record_fail("Hamiltonian construction", str(exc))

# ============================================================================
# TEST 5: Time Evolution (Simple Decay)
# ============================================================================
print("\n[TEST 5] Time Evolution - Simple Decay")
try:
    N = 5
    a = destroy(N)
    w = 1.0
    gamma = 0.1

    H = w * a.dag() * a
    c_ops = [np.sqrt(2 * gamma) * a]
    psi0 = basis(N, 1)

    times = np.linspace(0, 10, 100)
    result = mesolve(H, psi0, times, c_ops, [a.dag() * a])

    if not np.isclose(result.expect[0][0], 1.0, atol=1e-6):
        results.record_fail("Initial population", f"⟨n⟩(0) = {result.expect[0][0]}")
    else:
        results.record_pass("Initial population")

    t_half = times[len(times) // 2]
    n_expected = np.exp(-2 * gamma * t_half)
    n_actual = result.expect[0][len(times) // 2]
    if not np.isclose(n_actual, n_expected, rtol=0.2):
        results.record_fail(
            "Exponential decay",
            f"⟨n⟩({t_half:.2f}) = {n_actual:.3f}, expected {n_expected:.3f}",
        )
    else:
        results.record_pass("Exponential decay")

    if result.expect[0][-1] > 0.2:
        results.record_fail("Decay to ground state", f"Final ⟨n⟩ = {result.expect[0][-1]:.3f}")
    else:
        results.record_pass("Decay to ground state")
except Exception as exc:  # pragma: no cover - defensive
    results.record_fail("Time evolution", str(exc))

# ============================================================================
# TEST 6: Markovian Model (Full System)
# ============================================================================
print("\n[TEST 6] Markovian Model - Full System")
try:
    times = np.linspace(0, 0.2, 60)
    times_m, x3_m, n3_m = run_markovian_model(N=6, times=times, excitation_amp=0.5)

    if len(x3_m) != len(times_m) or len(n3_m) != len(times_m):
        results.record_fail("Result length", "Outputs do not match requested times")
    else:
        results.record_pass("Result structure")

    if not np.all(np.isreal(x3_m)):
        results.record_fail("Position trajectory reality", "x3(t) has imaginary parts")
    else:
        results.record_pass("Position trajectory reality")

    if np.any(n3_m < -1e-10):
        results.record_fail("Energy positivity", f"Min ⟨n₃⟩ = {np.min(n3_m)}")
    else:
        results.record_pass("Energy positivity")

    if np.max(np.abs(x3_m)) < 1e-8:
        results.record_fail("Energy transfer", "Slow-layer displacement below tolerance")
    else:
        results.record_pass("Energy transfer to slow layer")
except Exception as exc:  # pragma: no cover - defensive
    results.record_fail("Markovian model", str(exc))

# ============================================================================
# TEST 7: Pseudomode Model Construction
# ============================================================================
print("\n[TEST 7] Pseudomode Model Construction")
try:
    N = 8
    N_pseudo = 6

    a1 = tensor(destroy(N), qeye(N), qeye(N), qeye(N_pseudo))
    a2 = tensor(qeye(N), destroy(N), qeye(N), qeye(N_pseudo))
    a3 = tensor(qeye(N), qeye(N), destroy(N), qeye(N_pseudo))
    a_pseudo = tensor(qeye(N), qeye(N), qeye(N), destroy(N_pseudo))

    expected_dim = N**3 * N_pseudo
    if a1.shape != (expected_dim, expected_dim):
        results.record_fail(
            "Pseudomode dimensions",
            f"Expected ({expected_dim},{expected_dim}), got {a1.shape}",
        )
    else:
        results.record_pass("Four-oscillator tensor product")

    w1, w2, w3 = 1000.0, 10.0, 1.0
    g12, g23 = 0.05, 0.05
    gamma1 = 0.1
    tau_B = 1e-3
    w_pseudo = w1
    kappa = 1.0 / tau_B
    g_pseudo = np.sqrt(kappa * gamma1 * w1 / np.pi)

    H = w1 * a1.dag() * a1 + w2 * a2.dag() * a2 + w3 * a3.dag() * a3
    H += w_pseudo * a_pseudo.dag() * a_pseudo
    H += g12 * (a1.dag() * a2 + a1 * a2.dag())
    H += g23 * (a2.dag() * a3 + a2 * a3.dag())
    H += g_pseudo * (a1.dag() * a_pseudo + a1 * a_pseudo.dag())

    if not H.isherm:
        results.record_fail("Pseudomode Hamiltonian", "Not Hermitian")
    else:
        results.record_pass("Pseudomode Hamiltonian Hermiticity")

    if g_pseudo < 0 or g_pseudo > 1000:
        results.record_fail("Pseudomode coupling", f"g_pseudo = {g_pseudo:.2f} unreasonable")
    else:
        results.record_pass("Pseudomode coupling magnitude")

    psi0 = tensor(basis(N, 0), basis(N, 0), basis(N, 0), basis(N_pseudo, 0))
    E0 = expect(H, psi0)
    if not np.isclose(E0, 0.0, atol=1e-10):
        results.record_fail("Pseudomode ground state", f"E0 = {E0}")
    else:
        results.record_pass("Pseudomode ground state energy")
except Exception as exc:  # pragma: no cover - defensive
    results.record_fail("Pseudomode model", str(exc))

# ============================================================================
# TEST 8: Short Pseudomode Evolution
# ============================================================================
print("\n[TEST 8] Pseudomode Evolution (Short Run)")
try:
    times = np.linspace(0, 0.3, 40)
    _, x3_s, n3_s = run_pseudomode_model(
        tau_B=1e-3,
        N=4,
        N_pseudo=4,
        times=times,
        excitation_amp=0.5,
    )

    if np.any(np.isnan(x3_s)) or np.any(np.isinf(x3_s)):
        results.record_fail("Pseudomode evolution stability", "NaN or Inf detected")
    else:
        results.record_pass("Pseudomode evolution stability")

    if np.max(np.abs(x3_s)) < 1e-8:
        results.record_fail("Pseudomode energy transfer", "Displacement below tolerance")
    else:
        results.record_pass("Pseudomode energy transfer")
except Exception as exc:  # pragma: no cover - defensive
    results.record_fail("Pseudomode evolution", str(exc))

# ============================================================================
# TEST 9: Power Spectrum Computation
# ============================================================================
print("\n[TEST 9] Power Spectrum Computation")
try:
    t = np.linspace(0, 10, 1000)
    f_test = 5.0
    x_test = np.sin(2 * np.pi * f_test * t)

    freq, psd = signal.periodogram(x_test, fs=1 / (t[1] - t[0]))
    peak_freq = freq[np.argmax(psd)]
    if not np.isclose(peak_freq, f_test, rtol=0.1):
        results.record_fail("Power spectrum", f"Peak at {peak_freq:.2f} Hz, expected {f_test:.2f} Hz")
    else:
        results.record_pass("Power spectrum computation")

    psd_norm = psd / np.max(psd)
    if not np.isclose(np.max(psd_norm), 1.0):
        results.record_fail("PSD normalization", "Max != 1.0 after normalization")
    else:
        results.record_pass("PSD normalization")
except Exception as exc:  # pragma: no cover - defensive
    results.record_fail("Power spectrum", str(exc))

# ============================================================================
# TEST 10: Memory Usage Estimate
# ============================================================================
print("\n[TEST 10] Memory Usage Check")
try:
    N = 15
    N_pseudo = 10

    dim = N**3 * N_pseudo
    bytes_per_complex = 16  # complex128
    dm_size_bytes = dim**2 * bytes_per_complex
    dm_size_MB = dm_size_bytes / (1024**2)

    print(f"  Hilbert space dimension: {dim}")
    print(f"  Density matrix size:    {dm_size_MB:.1f} MB")

    if dm_size_MB > 12_000:
        print("  Warning: large density matrix; ensure sufficient RAM before full runs.")
        results.record_pass(f"Memory usage caution ({dm_size_MB:.0f} MB)")
    else:
        results.record_pass(f"Memory usage ({dm_size_MB:.0f} MB)")
except Exception as exc:  # pragma: no cover - defensive
    results.record_fail("Memory check", str(exc))

# ============================================================================
# TEST 11: Parameter Sanity Checks
# ============================================================================
print("\n[TEST 11] Parameter Sanity Checks")
try:
    w1, w2, w3 = 1000.0, 10.0, 1.0
    g12, g23 = 0.05, 0.05

    if not (w1 > w2 > w3):
        results.record_fail("Frequency hierarchy", f"w1={w1}, w2={w2}, w3={w3} not ordered")
    else:
        results.record_pass("Frequency hierarchy")

    epsilon12 = g12 / w1
    epsilon23 = g23 / w2
    if epsilon12 > 0.1 or epsilon23 > 0.1:
        results.record_fail(
            "Weak coupling assumption",
            f"ε12={epsilon12:.3f}, ε23={epsilon23:.3f} > 0.1",
        )
    else:
        results.record_pass("Weak coupling regime")

    tau_B_test = 1e-3
    theta = w1 * tau_B_test
    if theta < 0.1 or theta > 10:
        results.record_fail("Bath parameter Θ", f"Θ = {theta:.2f} outside [0.1, 10]")
    else:
        results.record_pass("Bath correlation parameter Θ")
except Exception as exc:  # pragma: no cover - defensive
    results.record_fail("Parameter checks", str(exc))

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 60)
success = results.summary()
print("=" * 60)

if success:
    print("\n🚀 System is ready! You can now run:")
    print("   python quantum_hierarchy.py")
else:
    print("\n⚠️  Please fix the failed tests before proceeding.")
    time.sleep(0.25)
    sys.exit(1)
