Adaptive re-runs (supplementary)
================================

This folder contains small, targeted re-runs used to confirm the same diagnostics with tighter sampling and short runtime. There is **no change in model** or physics; only runtime and sampling refinements (smaller Hilbert sizes, shorter traces, and early-stop heuristics).

Files
-----
- `classC_fraction_attempt.csv`
  - Class C (parametric modulation) with modulation at the fast scale (fractional units of `W1`).
  - Shows clear PSD surrogate failure (PSD-NRMSE > 1) and band-ratio separation across Θ.
- `kerr08_det12_fast3.csv`
  - Class M (equal-carrier, detuned + Kerr) fast two-point confirmation (Θ ∈ {1.0, 1.1}).
  - Peak R_env ≈ 1.06 with |ΔJ|/J* ≲ 1e-3.
- `kerr08_det12_theta_grid.csv`
  - Class M fast four-point grid (Θ ∈ {0.9, 1.0, 1.1, 1.2}); peak R_env ≈ 1.11 near Θ ≈ 0.9.

Reproduce (commands)
--------------------
All commands executed with early-stop enabled and limited thread counts.

Class C (fractional modulation):

```
python classical_parametric_sweep.py \
  --output results/adaptive/classC_fraction_attempt.csv \
  --mod-amp 0.3 --n-seeds 4 --duration 25.0 \
  --welch-nperseg 4096 --welch-noverlap 2048 \
  --early-stop --early-psd-threshold 0.6 --early-gap-threshold 0.1
```

Class M (fast grids):

```
python quantum_nonlin_sweep.py \
  --output results/adaptive/kerr08_det12_fast3.csv \
  --theta-list 1.0 1.1 --detune-frac 0.12 --kerr-fast 0.08 \
  --t-final 2.0 --n-time 1400 --burn-frac 0.4 \
  --excitation-amp 0.06 --n-hierarchy 4 --n-pseudo 3 \
  --equal-carrier-tol 0.001 --a1-init-alpha 0.2

python quantum_nonlin_sweep.py \
  --output results/adaptive/kerr08_det12_theta_grid.csv \
  --theta-list 0.9 1.0 1.1 1.2 --detune-frac 0.12 --kerr-fast 0.08 \
  --t-final 2.0 --n-time 1400 --burn-frac 0.4 \
  --excitation-amp 0.06 --n-hierarchy 4 --n-pseudo 3 \
  --equal-carrier-tol 0.001 --a1-init-alpha 0.2
```

Notes
-----
- Equal-carrier gate remained tight (|ΔJ|/J* ≲ 1e-3) in Class M runs.
- These CSVs are intended as **supplementary confirmations**; the main production datasets remain authoritative.

