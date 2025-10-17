# MRL Synthesis Paper (Overleaf-ready)

This archive contains a compilable LaTeX project that adopts the synthesis framing and the practical-equivalence gates (Option B).

## Files
- `main.tex` — full manuscript
- `references.bib` — bibliography (based on your provided refs)
- `figures/figA_classical.pdf` — placeholder (replace with production figure)
- `figures/figB_equal_carrier.pdf` — placeholder
- `figures/figC_robustness.pdf` — placeholder

## Notes
- Config hash macro: `\confighash = c7dc5aa1` (short form). Update if you prefer the long hash.
- Gates (classical): PSD-NRMSE < 0.03 **and** |d_z| < 0.30; Holm p reported, not gated.
- Quantum: equal-carrier tolerance |ΔJ|/J* ≤ 0.02; j_bandwidth_frac = 0; parity test ≤ 1e-3 at Θ = 0.95.
- Replace the placeholder PDFs with your generated `figA_classical.pdf`, `figB_equal_carrier.pdf`, `figC_robustness.pdf` and recompile.

## How to use
1. Download the ZIP.
2. Upload to Overleaf as a new project.
3. Replace placeholder figures with your production PDFs.
4. Compile (pdfLaTeX).
