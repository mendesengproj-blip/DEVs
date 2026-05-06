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

- DEV action and DBI kinetic term
- SPARC fit (167 galaxies, χ²ᵥ = 1.20)
- Cosmological consistency via fσ₈ (Δχ² = -0.22 vs ΛCDM)
- Falsifiable UDG slip prediction (2-10%, Euclid-detectable)
- Bullet Cluster order-of-magnitude estimate

📁 Code: [`paper_I/`](paper_I/)

### Paper II — Stability and Robustness
**"Stability, Scale Constraints, and Parameter Robustness of
Vacuum Excitation Dynamics"**

- Sound speed: c²ₛ ∈ [1/3, 1) — stable everywhere
- Vector rigidity scale: L < 17 pc from SPARC
- β–Υ orthogonality and jackknife robustness (1.3% spread)

📁 Code: [`paper_II/`](paper_II/)

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
├── paper_I/          # Foundational paper code
├── paper_II/         # Stability follow-up code
├── README.md         # This file
├── .gitignore
├── requirements.txt
└── .gitmessage       # Commit message template
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
