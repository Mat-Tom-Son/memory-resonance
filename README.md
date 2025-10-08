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

## Researcher’s Guide

**Bootstrapping the Environment**
- `python -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`
- `python test_quantum_hierarchy.py` (≈20 s). If anything fails, read the per-test notes at the bottom; rerun after fixes.
- `python quick_diagnostic.py` is the fastest sanity check after changing QuTiP or numpy.

**Core Quantum Workflow**
- `python stage1_markovian.py` (≈30 s). Confirms the Markovian baseline and produces `markovian_baseline.png`.
- `python stage2_pseudomode.py` (≈1 min). Gives Markovian vs. structured overlays (`quantum_comparison.png`) plus mean-energy and band-power ratios in the terminal.
- `python stage3_parameter_sweep.py` (a few minutes). Generates the τ_B sweep (`parameter_sweep.png`). Adjust the `tau_B_values` grid in the script before running if you need different coverage.
- `python quantum_hierarchy.py` (4–6 min with defaults). End-to-end run; prints amplification metrics and writes `quantum_envelope_mechanism.png`.
  - To change excitation strength or Hilbert cutoffs, pass arguments through `run_markovian_model(..., excitation_amp=…)` or `run_pseudomode_model(N=…, N_pseudo=…)`. Scripts import these helpers, so edit the call sites near the top of each file.

**High-Fidelity Classical Benchmark**
- Open `hierarchical_analysis.py`, set `mode = 'equal_variance'` or `'equal_carrier'`.
- Run `python hierarchical_analysis.py` (10–20 min). You’ll see:
  1. Stage banners for calibration mode.
  2. Progress through the controlled comparison.
  3. A tqdm bar during the τ_B sweep (25 points × 16 seeds by default).
- Outputs: `experiment1_<mode>.png`, `experiment2_sweep_<mode>.png`, and a terminal summary with bootstrap CI on the Θ peak.
- To shorten exploratory runs, lower `n_seeds`, shrink `tau_B_values`, or reduce `T`. Revert to defaults before final figures.

**Performance Planning**
- `python benchmark.py` samples three Hilbert configurations and extrapolates runtime for the full pseudomode simulation. Update the `configs` list with your planned `N`, `N_pseudo` values before launching a long job.
- For quantum runs, keep density matrices under ~20 GB unless you have plenty of RAM (see the warning in `test_quantum_hierarchy.py`).

**Custom Experiments**
- Add new observables in `quantum_models.py` (e.g., quadratures or entanglement entropy) and return them alongside `x₃, n₃`. Propagate them into the stage scripts to plot.
- Modify `stage3_parameter_sweep.py` to scan other parameters (couplings, damping) by editing the loop at the top; reuse the PSD helper for consistent band-power metrics.
- In the classical script, swap `dt`, `T`, or the PSD band to align with specific experimental conditions. The bootstrap utilities (`bootstrap_ci`) can wrap any statistic you care about.

**Reproducibility Checklist**
1. Record the git commit hash and `pip freeze` output after installing requirements.
2. Ensure `python test_quantum_hierarchy.py` passes.
3. Run the relevant stage scripts or `quantum_hierarchy.py` with clearly noted parameter changes.
4. Capture `hierarchical_analysis.py` outputs for the same calibration mode.
5. Archive the generated figures and terminal logs together with your report/notebook.

### Troubleshooting Tips

- Lower the Fock cutoffs (e.g., `N=10`) or shorten the time grids if simulations are slow.
- Increase the initial excitation amplitude for clearer envelopes when τ_B is near the threshold regime.
- Ensure every tensor product includes the correct number of identity factors if you modify the Hilbert space layout.
