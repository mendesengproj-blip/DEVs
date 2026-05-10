# DEVs — Vacuum Excitation Dynamics

Scalar-vector-tensor effective field theory of dark matter
as a phase transition of the quantum vacuum.

**Author:** Miqueias Alves Mendes
**Institution:** Independent Researcher, Ibiapina, Ceará, Brazil
**Repository:** github.com/mendesengproj-blip/DEVs

---

## Series status

| Paper | Title | File | Status |
|-------|-------|------|--------|
| I | Vacuum Excitation Dynamics: A SVT Field Theory of Dark Matter as a Phase Transition of the Quantum Vacuum | `paper_I/dev_paper_I_final.tex` | Submitted to Physical Review D (DS14085) |
| II | Stability, Scale Constraints, and Parameter Robustness of Vacuum Excitation Dynamics | `paper_II/dev_paper_II_FINAL.tex` | Ready for JCAP submission |
| III | Non-Local Gravitational Slip in Vacuum Excitation Dynamics: Extended-Source Derivation | `paper_III/DEV_paper_III_FINAL.tex` | Ready for submission |

---

## Key results

- **SPARC fits:** χ²ν = 1.20 on 167 galaxies, zero global free parameters
- **Gravitational slip:** η−1 ∈ [2.2%, 4.1%] (point-source) and [3.8%, 6.9%] (extended-source) for six benchmark UDGs
- **Stability:** cs² ∈ [1/3, 1), ghost-free, L < 17 pc, mA > 3.7×10⁻²⁵ eV/c²
- **Non-local operator:** α = −1.56 ± 0.02 (deep-MOND), regime-dependent (α → −2 in Newton regime)
- **Euclid forecast:** SNR ~31σ for N=300 UDGs at σ_η = 0.02
- **Falsification criterion:** η−1 < 1% across UDG sample rules out DEV

---

## Repository structure

```
DEVs/
├── paper_I/
│   ├── dev_paper_I_final.tex      # Paper I — final version (PRD DS14085)
│   ├── theory.py                  # DEV field equations and ν function
│   ├── sparc.py                   # SPARC rotation curve fitting pipeline
│   ├── udg.py                     # UDG gravitational slip predictions
│   ├── calibrate_beta.py          # β calibration from lensing constraints
│   └── figures/                   # All Paper I figures
│
├── paper_II/
│   ├── dev_paper_II_FINAL.tex     # Paper II — final version (JCAP ready)
│   ├── stability.py               # cs² and ghost-freedom analysis
│   ├── vector_scale.py            # L bound from SPARC inner points
│   ├── degeneracies.py            # β/Υ★ Fisher matrix analysis
│   ├── beta_naturalness.py        # β scale-invariance test
│   └── figures/                   # All Paper II figures
│
├── paper_III/
│   ├── DEV_paper_III_FINAL.tex    # Paper III — final version
│   ├── operator_identification.py # Inverse-path α identification
│   ├── universality_test.py       # α universality test (4 profiles)
│   ├── operator_identification.png
│   ├── operator_comparison.png
│   ├── universality_Seff.png
│   └── universality_alpha_vs_epsilon.png
│
├── code/
│   └── verification/
│       ├── alpha_sympy_verification.py   # α=2/3 symbolic proof
│       ├── alpha_sympy_report.txt
│       ├── eta_diagnosis.py              # η non-locality diagnosis
│       ├── eta_diagnosis_report.txt
│       └── eta_verification_report.txt
│
├── dev_refs.bib                   # Complete bibliography (21 references)
├── DEV_series_status.md           # Series audit report
├── CLAUDE.md                      # Context for Claude Code sessions
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Canonical parameters

| Parameter | Value |
|-----------|-------|
| a₀ | 1.2×10⁻¹⁰ m/s² = 3703 (km/s)²/kpc |
| β | 0.0075 [0.0015, 0.0134] |
| G | 4.302×10⁻³ kpc·(km/s)²·Msun⁻¹ |
| c | 2.998×10⁵ km/s |

---

## Reproducing the results

```bash
pip install -r requirements.txt
cd paper_I && python sparc.py        # reproduces χ²ν = 1.20
cd paper_I && python udg.py          # reproduces Table III
cd paper_III && python operator_identification.py  # reproduces α = -1.56
cd paper_III && python universality_test.py        # reproduces Table II
```

---

## Citation

If you use this work, please cite:

Mendes, M. A. (2026). *Vacuum Excitation Dynamics: A
Scalar-Vector-Tensor Field Theory of Dark Matter as a
Phase Transition of the Quantum Vacuum*.
Submitted to Physical Review D, manuscript DS14085.
