# Memory-Resonance Condition: Cross-Domain Control for Colored-Noise Systems

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](mrl_synthesis_paper/main.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Preprint-orange)](PAPER_STATUS.md)

This repository contains the simulation code, data, and analysis for our paper on the **Memory-Resonance Condition (MRC)**: a cross-domain design principle showing that systems across classical and quantum domains exhibit optimal performance when environmental memory synchronizes with internal dynamics (Θ ≡ ω_fast τ_B ≈ 1).

---

## Quick Links

- **Paper:** [`mrl_synthesis_paper/main.pdf`](mrl_synthesis_paper/main.pdf) (390 KB, 11 pages)
- **Data:** [`results/theta_sweep_today.csv`](results/theta_sweep_today.csv) (301 KB consolidated dataset)
- **Figures:** [`figures/`](figures/) (3 publication-ready PDFs)
- **Status:** [`PAPER_STATUS.md`](PAPER_STATUS.md) (current version info)

---

## What is the MRC?

**Memory-Resonance Condition:** Systems often perform best when their environment's correlation time (τ_B) matches their fastest internal rhythm (1/ω_fast):

**Θ ≡ ω_fast τ_B ≈ 1**

**Key Insight:** The *observable* (a shallow optimum near Θ≈1) recurs across substrates—from stochastic resonance in neural circuits to noise-assisted quantum transport—but the *mechanism* varies:

- **Class S (Spectral):** Frequency-domain overlap in near-linear systems
- **Class C (Coherent):** Weak nonlinearity reweights spectra
- **Class M (Memory):** Time-nonlocal dynamical kernels

**Our Contribution:** We formalize this cross-domain pattern, provide operational diagnostics to classify mechanism, and validate with rigorous controls across the classical/quantum divide.

---

## Repository Structure

```
.
├── mrl_synthesis_paper/          # Paper source (LaTeX + PDF)
│   ├── main.tex                  # Main paper source
│   ├── main.pdf                  # Compiled PDF (390 KB, 11 pages)
│   └── references.bib            # Bibliography
│
├── figures/                      # Figure generation scripts
│   ├── make_figA.py             # Classical pillar (OU vs surrogate)
│   ├── make_figB.py             # Quantum pillar (equal-carrier scan)
│   ├── make_figC.py             # Robustness (metric consistency)
│   ├── figA_classical.pdf       # Generated figure A
│   ├── figB_equal_carrier.pdf   # Generated figure B
│   └── figC_robustness.pdf      # Generated figure C
│
├── results/                      # Simulation data and manifests
│   ├── theta_sweep_today.csv    # Consolidated dataset (301 KB)
│   └── production_archive/      # QA logs, manifests, config hash
│
├── src/                         # Core simulation code
│   ├── quantum_models.py        # Quantum hierarchy (Lindblad + pseudomode)
│   ├── hierarchical_analysis.py # Classical analysis (OU bath, controls)
│   └── ...                      # Supporting modules
│
├── tests/                       # Validation suite
│   └── test_quantum_hierarchy.py
│
├── PAPER_STATUS.md              # Current paper status
├── SUBMISSION_READY.md          # Pre-submission checklist
└── README.md                    # This file
```

---

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/Mat-Tom-Son/memory-resonance.git
cd memory-resonance

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Reproduce Paper Figures

```bash
# Regenerate all three figures from consolidated data
cd figures
PYTHONPATH=.. python make_figA.py  # Classical pillar (~5 sec)
PYTHONPATH=.. python make_figB.py  # Quantum pillar (~5 sec)
PYTHONPATH=.. python make_figC.py  # Robustness (~5 sec)
```

**Output:** `figA_classical.pdf`, `figB_equal_carrier.pdf`, `figC_robustness.pdf`

### 3. Run Tests

```bash
# Validate simulation code
pytest tests/test_quantum_hierarchy.py -v

# Quick diagnostic (10 seconds)
python quick_diagnostic.py
```

---

## Key Results

### Classical Pillar (Class S - Spectral Overlap)

**Finding:** Ornstein-Uhlenbeck (OU) noise and PSD-matched surrogates are practically equivalent.

**Gates:** PSD-NRMSE < 0.03, |d_z| < 0.30 ✓

**Interpretation:** Classical Θ-dependence arises from spectral overlap (Wiener-Khinchin), consistent with stochastic resonance literature.

### Quantum Pillar (Class M - Memory Backaction)

**Finding:** Equal-carrier scans retain Θ≈1 structure despite fixed spectral weight at ω₁.

**Gate:** |ΔJ|/J* ≤ 0.02 ✓

**Interpretation:** Residual Θ-structure with controlled spectral weight confirms memory backaction mechanism, transcending the classical/quantum divide.

### Robustness

**Finding:** Baseband and narrowband metrics agree in ordering across Θ.

**Interpretation:** MRC is a system-environment property, not a metric-specific artifact.

---

## Reproduce from Scratch

### Classical Analysis (Long Run)

```bash
# Full OU calibration + surrogate comparison + Θ sweep
python hierarchical_analysis.py  # ~20 minutes, 25 Θ points × 16 seeds

# Outputs:
# - experiment1_equal_variance.png
# - experiment2_sweep_equal_variance.png
# - Terminal: Bootstrap CI on Θ peak
```

### Quantum Hierarchy

```bash
# Stage 1: Markovian baseline
python stage1_markovian.py  # ~30 sec

# Stage 2: Pseudomode comparison
python stage2_pseudomode.py  # ~1 min

# Stage 3: τ_B parameter sweep
python stage3_parameter_sweep.py  # ~5 min

# Or run complete pipeline:
python quantum_hierarchy.py  # ~6 min
```

---

## Paper Highlights

### Design Card (Actionable Guidance)

The paper includes a boxed **Design Card** with:

1. **Rule:** Target Θ in the MR band [0.7, 1.4]
2. **How to estimate ω_fast:** 3 cases (linear, nonlinear, driven)
3. **How to choose τ_B:** Use τ_B^(eff) (observable-effective timescale)
4. **Diagnostics:** Tests to classify Classes S/C/M
5. **Failure modes:** When MRC may not apply
6. **Optional controller:** Two-point dither for τ_B adaptation

### Synthesis Map (Cross-Domain Evidence)

The paper synthesizes evidence from:

- **Stochastic resonance** (Mondal et al. 2018, Gammaitoni et al.)
- **Coherence resonance** (Brugioni et al. 2005, Pikovsky & Kurths)
- **Quantum transport** (Moreira et al. 2020, Plenio & Huelga)
- **Photosynthesis** (Uchiyama et al. 2017)
- **Neural detection** (Duan et al. 2014)
- **Energy harvesting** (Romero-Bastida & López 2020)

---

## Data Availability

All simulation code, raw data, configuration manifests, and figure-generation scripts are in this repository:

- **Consolidated dataset:** `results/theta_sweep_today.csv` (301 KB)
- **Config hash:** `c7dc5aa1` (production run identifier)
- **Reproduction scripts:** `figures/make_fig*.py`
- **QA manifests:** `results/production_archive/`

**Reproducibility:** All figures can be regenerated from source data. Gates, seeds, and quality checks are documented.

---

## Advanced Usage

### Custom Parameter Sweeps

```python
# Modify stage3_parameter_sweep.py
tau_B_values = np.linspace(0.1, 3.0, 50)  # Denser Θ grid
# Then run: python stage3_parameter_sweep.py
```

### Custom Observables

```python
# Add to quantum_models.py
def compute_entanglement_entropy(rho, subsystem_dims):
    # Your implementation
    return S_ent

# Use in stage scripts to plot new metrics
```

### Performance Tuning

```bash
# Estimate runtime for large Hilbert spaces
python benchmark.py

# For classical runs, reduce n_seeds or T for exploration:
# In hierarchical_analysis.py:
n_seeds = 8       # Default: 16
T = 100.0         # Default: 200.0
```

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{Thompson2025MRC,
  title={Memory-Resonance Condition: A Cross-Domain Control Principle for Colored-Noise Systems},
  author={Thompson, Mat},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

*(Update arXiv number once preprint is live)*

---

## Contact & Contributing

- **Issues:** [GitHub Issues](https://github.com/Mat-Tom-Son/memory-resonance/issues)
- **Questions:** Open a discussion or email [mat.tom.son@protonmail.com](mailto:mat.tom.son@protonmail.com)
- **Contributing:** Pull requests welcome!

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [QuTiP](https://qutip.org/) for quantum simulation framework
- All cited authors for cross-domain evidence synthesis
- Reviewers and collaborators for feedback

---

## Additional Documentation

- **Paper Status:** [`PAPER_STATUS.md`](PAPER_STATUS.md) - Current version, stats, checklist
- **Submission Ready:** [`SUBMISSION_READY.md`](SUBMISSION_READY.md) - Final checks, talking points
- **Session Notes:** `.claude_session_notes/` - Detailed revision history (hidden)

---

**Status:** Preprint ready for submission (2025-10-12)

**Config hash:** c7dc5aa1 | **PDF:** 390 KB, 11 pages | **Data:** 301 KB CSV

