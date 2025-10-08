"""
Quick diagnostic to verify QuTiP evolution in ~10 seconds.
"""

from __future__ import annotations

import time

import numpy as np
from qutip import basis, destroy, mesolve


def main():
    print("Quick Diagnostic Test (~10 seconds)")
    print("=" * 40)

    start = time.time()

    N = 8
    a = destroy(N)
    H = a.dag() * a
    c_ops = [np.sqrt(0.1) * a]
    psi0 = basis(N, 1)
    times = np.linspace(0, 5, 50)

    print("Running minimal evolution...")
    result = mesolve(H, psi0, times, c_ops, [a.dag() * a])

    elapsed = time.time() - start

    n_initial = result.expect[0][0]
    n_final = result.expect[0][-1]
    decayed = n_initial > n_final

    print("\nResults:")
    print(f"  ⟨n⟩(0) = {n_initial:.3f}")
    print(f"  ⟨n⟩(T) = {n_final:.3f}")
    print(f"  Decayed? {decayed}")
    print(f"  Time: {elapsed:.2f}s")

    if decayed and 0.9 < n_initial < 1.1 and n_final < 0.5:
        print("\n✓ System working correctly")
    else:
        print("\n✗ Something is wrong!")


if __name__ == "__main__":
    main()
