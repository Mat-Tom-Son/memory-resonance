"""
Benchmark script estimating runtime for different Hilbert space sizes.
"""

from __future__ import annotations

import time

import numpy as np
from qutip import basis, destroy, mesolve, qeye, tensor


def benchmark_size(N: int, N_pseudo: int) -> float:
    """Run a tiny pseudomode evolution and return wall-clock time."""
    a1 = tensor(destroy(N), qeye(N), qeye(N), qeye(N_pseudo))
    a2 = tensor(qeye(N), destroy(N), qeye(N), qeye(N_pseudo))
    a3 = tensor(qeye(N), qeye(N), destroy(N), qeye(N_pseudo))
    a_pseudo = tensor(qeye(N), qeye(N), qeye(N), destroy(N_pseudo))

    w1, w2, w3 = 1000.0, 10.0, 1.0
    g12, g23 = 0.05, 0.05
    gamma1, gamma2, gamma3 = 0.1, 0.05, 0.01
    tau_B = 1e-3

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
    times = np.linspace(0, 0.1, 10)  # short evolution for timing

    start = time.time()
    mesolve(H, psi0, times, c_ops, [a3.dag() * a3])
    elapsed = time.time() - start
    return elapsed


def main():
    print("Benchmarking different Hilbert space sizes...")
    print("=" * 60)

    configs = [
        (8, 6, "Fast (testing)"),
        (10, 8, "Medium (draft)"),
        (15, 10, "Full (publication)"),
    ]

    for N, N_pseudo, label in configs:
        dim = N**3 * N_pseudo
        print(f"\n{label}: N={N}, N_pseudo={N_pseudo}")
        print(f"  Hilbert dimension: {dim}")

        try:
            timing = benchmark_size(N, N_pseudo)
            # Scale to a typical run (2.0 time units, 1000 steps)
            scaled = timing * (1000 / 10) * (2.0 / 0.1)
            print(f"  Test time:          {timing:.2f}s")
            print(f"  Estimated full run: {scaled:.1f}s ({scaled/60:.1f} min)")
        except MemoryError:
            print("  ✗ Out of memory!")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  ✗ Error: {exc}")

    print("\n" + "=" * 60)
    print("Recommendation: choose the largest configuration under ~5 minutes.")


if __name__ == "__main__":
    main()

