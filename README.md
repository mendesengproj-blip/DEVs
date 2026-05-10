# Vacuum Excitation Dynamics (DEV)

**A Scalar-Vector-Tensor Field Theory of Dark Matter as a Phase Transition
of the Quantum Vacuum**

*Miqueias Alves Mendes — Independent Researcher, 2026*

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

DEV proposes that dark matter is not a primordial particle species but a
**phase transition of the gravitational vacuum**, triggered when the
Newtonian acceleration falls below the critical scale
a₀ ≃ 1.2×10⁻¹⁰ m/s².

This repository contains the complete analysis pipeline for the
DEV theory papers.

## Papers in this Series

### Paper I — Foundational Theory
**"Vacuum Excitation Dynamics: A Scalar-Vector-Tensor Field Theory of
Dark Matter as a Phase Transition of the Quantum Vacuum"**

> **Status:** Submitted to *Physical Review D* on 2026-05-08.
> Manuscript ID: **DS14085** — with editors.
> (MNRAS MN-26-1316-P rejected on scope, 2026-05-05.)

- DEV action and DBI kinetic term
- SPARC fit (167 galaxies, χ²ᵥ = 1.20)
- Cosmological consistency via fσ₈ (Δχ² = -0.22 vs ΛCDM)
- Falsifiable UDG slip prediction (2-10%, Euclid-detectable)
- Bullet Cluster order-of-magnitude estimate
- α = 2/3 derived analytically and verified symbolically (SymPy)
- Slip formula η−1 = (2/3)β/√[y(1+y)] — Green's-function solution
  for a point source in deep-MOND, valid for r ≫ r_MOND
  (verified for all 6 UDG targets; min r_eff/r_MOND = 4.56)

📁 Source: [`paper_I/`](paper_I/) — final submitted version:
[`paper_I/dev_paper_I_final.tex`](paper_I/dev_paper_I_final.tex)

### Paper II — Stability, Robustness, and Naturalness
**"Stability, Scale Constraints, and Parameter Robustness of
Vacuum Excitation Dynamics"**

- Sound speed: c²ₛ ∈ [1/3, 1) — subluminal, ghost-free
- Vector rigidity scale: L < 17 pc from SPARC → m_A > 3.7×10⁻²⁵ eV
- β–Υ★ orthogonality and jackknife robustness (1.3% spread)
- β scale-invariant over 3 decades in g/a₀ (slope = −0.005 ± 0.018)
- β ≈ (Ω_Λ)^(1/3) within 15% — flagged as numerical coincidence,
  not a derivation
- Dark-photon kinetic mixing ruled out (required ε ~ 0.09 ≫ 10⁻³ bound)

📁 Source: [`paper_II/`](paper_II/) — final version:
[`paper_II/dev_paper_II_FINAL.tex`](paper_II/dev_paper_II_FINAL.tex)

> **Status:** Ready for submission to *JCAP*. Awaiting Paper I arXiv
> number to update cross-reference.

### Paper III — Non-Local Gravitational Slip in Vacuum Excitation Dynamics: Extended-Source Derivation

**File:** [`paper_III/DEV_paper_III_FINAL.tex`](paper_III/DEV_paper_III_FINAL.tex)
**Status:** Ready for submission (pending Paper I arXiv number)

**Central result:** The gravitational slip operator of Paper I is
intrinsically non-local. Numerical inverse-path analysis yields an
effective source exponent α = −1.56 ± 0.02 in the deep-MOND regime,
inconsistent with the Poisson kernel (α = −2) and consistent with a
fractional Laplacian (−∇²)^{3/4} with Green function ∝ r^{−1/2}. The
exponent is regime-dependent (α varies from −1.96 in the Newtonian
regime to −1.41 for UDGs), consistent with the saturation axiom.
Extended-source corrections yield η−1 ∈ [3.78%, 6.90%] for the six
benchmark UDGs — all within Euclid sensitivity. First-principles
derivation deferred to Paper IV.

📁 Source: [`paper_III/`](paper_III/)

### Independent verifications

📁 [`code/verification/`](code/verification/) — symbolic and numerical
audits backing the analytical claims:

- `alpha_sympy_verification.py` — α = 2/3 exact via SymPy
- `eta_diagnosis.py` — dimensional analysis and regime diagnosis of
  the slip formula (confirms validity in the point-source deep-MOND limit)
- `eta_verification_report.txt` — extended-source verification report

## Installation

```bash
git clone https://github.com/mendesengproj-blip/DEVs.git
cd DEVs
pip install -r requirements.txt
```

## Data Source

SPARC rotation curves are not redistributed in this repository.
Download from: http://astroweb.cwru.edu/SPARC/

Place in `paper_I/sparc_data/` for use with the pipeline.

## Reproducing the Results

### Paper I (foundational)
```bash
cd paper_I
python run_analysis.py
```

### Paper II (stability)
```bash
cd paper_II
python stability.py
python vector_scale.py
python degeneracies.py
```

## Repository Structure

```
DEVs/
├── paper_I/              # Paper I — LaTeX, code, submitted version
│   ├── dev_paper_FINAL_v2.tex   # version submitted to MNRAS
│   ├── dev_refs.bib
│   ├── corrections_log.txt
│   ├── run_analysis.py, sparc.py, theory.py, udg.py, ...
│   └── figures/
├── paper_II/             # Paper II — in preparation
│   ├── dev_paper_II.tex
│   ├── stability.py, vector_scale.py, degeneracies.py
│   ├── beta_naturalness.py, beta_naturalness_section.tex
│   ├── beta_dimensional_table.txt, beta_naturalness_report.txt
│   ├── beta_scale_consistency.png
│   ├── bullet_appendix.tex
│   ├── udg_rmond_table.txt, table_udg_expanded.tex
│   └── figures/
├── code/verification/    # Independent symbolic/numerical audits
│   ├── alpha_sympy_verification.py + report
│   ├── eta_diagnosis.py + report
│   └── eta_verification_report.txt
├── README.md
├── .gitignore
├── requirements.txt
└── .gitmessage
```

## Citation

If you use this code, please cite:

Mendes, M. A. (2026). *Vacuum Excitation Dynamics: A Scalar-Vector-Tensor
Field Theory of Dark Matter as a Phase Transition of the Quantum Vacuum*.
arXiv:XXXX.XXXXX [gr-qc]

## License

MIT License

## Contact

Miqueias Alves Mendes
mendes.eng.proj@gmail.com
https://github.com/mendesengproj-blip/DEVs

---

*All code publicly available for full reproducibility.*
