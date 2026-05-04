# Vacuum Excitation Dynamics (DEV)

**A Scalar-Vector-Tensor Field Theory of Dark Matter as a Phase Transition
of the Quantum Vacuum**

*Miqueias Alves Mendes — Independent Researcher, 2026*

[![arXiv](https://img.shields.io/badge/arXiv-gr--qc-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

DEV proposes that dark matter is not a primordial particle species but a
**phase transition of the gravitational vacuum**, triggered when the
Newtonian acceleration falls below the critical scale
a₀ ≃ 1.2×10⁻¹⁰ m/s².

The theory is anchored by a Dirac-Born-Infeld (DBI) kinetic term that
reproduces the standard MOND interpolation function as its quasi-static
limit, and a massive vector field that generates a unique gravitational
slip signature detectable by Euclid.

## Key Results

| Result | Value |
|--------|-------|
| SPARC galaxies fitted | 167 |
| Median χ²_ν | 1.20 |
| Δχ² vs ΛCDM (fσ₈) | −0.22 (DEV marginally better) |
| Gravitational slip in UDGs | 2–10% |
| Fisher SNR (Euclid, N=300) | ~8σ |

## Installation

```bash
git clone https://github.com/mendesengproj-blip/DEVs.git
cd DEVs
pip install -r requirements.txt
```

## Data

SPARC rotation curves are **not included** in this repository due to
redistribution restrictions. Download from:

> Lelli, McGaugh & Schombert (2016), AJ 152, 157
> http://astroweb.cwru.edu/SPARC/

Place the downloaded `.dat` files in a folder named `sparc_data/`.

## Usage

```bash
# Run complete analysis
python run_analysis.py

# Fit SPARC catalog (requires sparc_data/ folder)
python sparc.py sparc_data/

# Calibrate β parameter
python calibrate_beta.py

# Generate all figures
python run_analysis.py
```

## Module Structure

| File | Description |
|------|-------------|
| `theory.py` | Core DEV equations: μ(x), ν(y), η(g), v_circ |
| `sparc.py` | SPARC data loader and rotation curve fitting |
| `udg.py` | Gravitational slip predictions for UDGs |
| `cosmology.py` | Modified growth equation and fσ₈ |
| `calibrate_beta.py` | β calibration from lensing constraints |
| `run_analysis.py` | Full analysis pipeline |

## Figures

All figures are pre-generated in `figures/` and correspond directly
to the paper.

## Citation

If you use this code, please cite:
Mendes, M. A. (2026). Vacuum Excitation Dynamics: A Scalar-Vector-Tensor
Field Theory of Dark Matter as a Phase Transition of the Quantum Vacuum.
arXiv:XXXX.XXXXX [gr-qc]

## License

MIT License — see LICENSE file.

## Contact

Miqueias Alves Mendes
mendes.eng.proj@gmail.com
https://github.com/mendesengproj-blip/DEVs

---

*Pipeline publicly available for full reproducibility.*
