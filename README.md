# Quantum Hierarchical Oscillator Experiment

This repository contains three incremental simulation stages plus a full-run script for exploring Markovian vs. non-Markovian damping in a hierarchy of coupled quantum oscillators. All models use [QuTiP](https://qutip.org/) for Lindblad evolution.

## Project Layout

- `quantum_models.py` – shared model helpers for Markovian and pseudomode dynamics.
- `stage1_markovian.py` – baseline Markovian simulation and figure.
- `stage2_pseudomode.py` – adds a pseudomode bath and compares spectra/energy.
- `stage3_parameter_sweep.py` – sweeps bath memory to map amplification.
- `quantum_hierarchy.py` – complete run producing all key metrics and figures.
- `hierarchical_analysis.py` – production classical experiment with OU bath calibration, Welch PSD, and sweep tooling.

Each script saves its primary figure in the working directory after execution.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running The Stages

```bash
python stage1_markovian.py           # Stage 1: Markovian baseline
python stage2_pseudomode.py          # Stage 2: pseudomode comparison
python stage3_parameter_sweep.py     # Stage 3: τ_B sweep and phase diagram
python quantum_hierarchy.py          # Complete simulation (Stages 1–3)
python hierarchical_analysis.py      # Classical production analysis (optional, long runtime)
```

## Verification & Diagnostics

```bash
python test_quantum_hierarchy.py     # Full validation suite (run first time)
python quick_diagnostic.py           # 10-second smoke test
python benchmark.py                  # Estimate runtime vs. Hilbert size
```

### Troubleshooting Tips

- Lower the Fock cutoffs (e.g., `N=10`) or shorten the time grids if simulations are slow.
- Increase the initial excitation amplitude for clearer envelopes when τ_B is near the threshold regime.
- Ensure every tensor product includes the correct number of identity factors if you modify the Hilbert space layout.
