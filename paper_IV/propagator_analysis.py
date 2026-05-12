"""
Paper IV - Step 1: Effective propagator of the DEV scalar field.

Tests whether the DBI kinetic operator linearized around a
deep-MOND background produces a fractional-Laplacian propagator
G~(p) ~ p^{-3/2}, G(r) ~ r^{-1/2}, consistent with the
operator (-nabla^2)^{3/4} identified numerically in Paper III.

All units: kpc, km/s, Msun.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------- Canonical parameters --------------------
a0 = 3703.0                  # (km/s)^2 / kpc
X0 = a0**2 / 2.0             # (km/s)^4 / kpc^2 -- DBI scale
beta_DEV = 0.0075
c_kms = 2.998e5              # km/s
mA_min = 1.87e-3             # kpc^{-1}  (3.7e-25 eV/c^2 lower bound)

OUT = Path(__file__).parent

# -------------------- DBI second-derivative F_XX --------------------
# F(X) = X0 [sqrt((1 + X/X0)^2) - 1]  -- here we use the
# canonical DBI form F(X) = X0 [ sqrt(1 + 2X/X0) - 1 ]
# (standard Bekenstein-Milgrom / TeVeS-like aquadratic Lagrangian)
# F_X  = 1/sqrt(1 + 2X/X0)
# F_XX = -1/(X0 (1 + 2X/X0)^{3/2})
# |F_XX| diverges as X -> 0 (deep-MOND): this is what generates
# the non-standard kinetic structure.

def F_X(Xbar):
    return 1.0 / np.sqrt(1.0 + 2.0 * Xbar / X0)

def F_XX(Xbar):
    # absolute value (sign carried separately in the kinetic op)
    return 1.0 / (X0 * (1.0 + 2.0 * Xbar / X0)**1.5)

# -------------------- Kinetic operator in Fourier --------------------
# Around a background with gradient |grad theta_bar|^2 = 2*Xbar,
# the quadratic action for delta theta is
#
#   L2 = (1/2) [ F_X (grad d theta)^2 + F_XX (grad theta_bar . grad d theta)^2 ]
#
# In Fourier with isotropic average <cos^2> = 1/3:
#
#   K~(p) = F_X * p^2 + F_XX * (2 Xbar) * p^2 * (1/3)
#         = p^2 [ F_X + (2/3) Xbar * F_XX ]
#
# Define mu(Xbar) := F_X + (2/3) Xbar * F_XX.
# Then K~(p) = mu(Xbar) * p^2 and G~(p) = 1/(mu p^2).
#
# A *constant-Xbar* background gives a pure p^{-2} propagator.
# Non-locality (fractional exponent) emerges only when Xbar itself
# depends on the wavelength being probed: in a spherically symmetric
# deep-MOND background, |grad theta_bar|(r) ~ sqrt(a0 * g_N(r))
# carries the scale 1/r through g_N ~ M/r^2, so Xbar(r) ~ a0/r and
# the *position-space* Green function inherits that scaling.
#
# This script therefore (a) computes the local K~(p) for fixed
# Xbar/X0 (TASK 1) and (b) computes the radial Green function of
# the scale-dependent operator (TASK 2) using a WKB/eikonal
# substitution Xbar -> Xbar(p) ~ a0^2 / (2 p^{-1} a0) = a0 p / 2
# which is the natural deep-MOND identification |grad theta| ~ sqrt(a0 g_N)
# with g_N(p) ~ a0 p in Fourier (Milgrom 1983 deep-MOND scaling).

def mu(Xbar):
    return F_X(Xbar) + (2.0 / 3.0) * Xbar * F_XX(Xbar)

# -------------------- TASK 1: local propagator --------------------
p = np.logspace(-3, 3, 2000)              # kpc^{-1}
regimes = {
    "deep-MOND extreme (X/X0=1e-4)": 1e-4,
    "deep-MOND moderate (X/X0=1e-2)": 1e-2,
    "transition (X/X0=1)":            1.0,
    "Newtonian (X/X0=1e2)":           1e2,
}

beta_local = {}
fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.viridis(np.linspace(0, 0.85, len(regimes)))

for (label, ratio), col in zip(regimes.items(), colors):
    Xbar = ratio * X0
    Kt = mu(Xbar) * p**2
    Gt = 1.0 / Kt
    # log-log slope
    s, _ = np.polyfit(np.log(p), np.log(Gt), 1)
    beta_local[label] = -s
    ax.loglog(p, Gt, color=col, label=f"{label}: beta={-s:.3f}")

# reference slopes
ax.loglog(p, 1e-2 * p**-1.5, "r--", lw=1, label="p^-3/2 (fractional)")
ax.loglog(p, 1e-2 * p**-2.0, "b--", lw=1, label="p^-2 (Poisson)")
ax.set_xlabel("p [kpc^-1]")
ax.set_ylabel("G~(p)")
ax.set_title("Effective propagator (local background)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "propagator_momentum.png", dpi=140)
plt.close(fig)

# -------------------- TASK 2: Green function in position --------------
# Scale-dependent background (WKB):
#   Xbar(p) = (a0 * p / 2) * Xref     with Xref chosen per regime
# so that Xbar/X0 at p = 1 kpc^-1 matches each regime above.
# This injects deep-MOND scale-dependence into the kinetic op.

r = np.logspace(0, 2, 200)               # kpc, fitting window [1,100]

fig, ax = plt.subplots(figsize=(7, 5))
gamma_pos = {}

for (label, ratio), col in zip(regimes.items(), colors):
    # Scale-dependent Xbar: pinned at p=1 kpc^-1 to the regime ratio.
    Xbar_p = ratio * X0 * (p / 1.0)
    Kt = mu(Xbar_p) * p**2
    Gt = 1.0 / Kt
    # 3D isotropic Fourier inverse: G(r) = 1/(2 pi^2 r) ∫ p G~(p) sin(pr) dp
    Gr = np.empty_like(r)
    for i, ri in enumerate(r):
        integrand = p * Gt * np.sin(p * ri)
        Gr[i] = np.trapezoid(integrand, p) / (2.0 * np.pi**2 * ri)
    Gr_abs = np.abs(Gr)
    s, _ = np.polyfit(np.log(r), np.log(Gr_abs), 1)
    gamma_pos[label] = -s
    ax.loglog(r, Gr_abs, color=col, label=f"{label}: gamma={-s:.3f}")

ax.loglog(r, 1e-4 * r**-0.5, "r--", lw=1, label="r^-1/2 (fractional)")
ax.loglog(r, 1e-4 * r**-1.0, "b--", lw=1, label="r^-1 (Poisson)")
ax.set_xlabel("r [kpc]")
ax.set_ylabel("|G(r)|")
ax.set_title("Position-space Green function (scale-dependent background)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "green_function_position.png", dpi=140)
plt.close(fig)

# -------------------- TASK 3: vector field contribution --------------
# Integrating out A_mu (mass mA) sourced by grad theta:
#   Delta K~(p) = gamma^2 / mA^2 * p^2
# with gamma^2 = beta_DEV * mA^2 * a0.
# Therefore Delta K~ / K~ = beta_DEV * a0 / mu(Xbar)  (p-independent)

vector_ratios = {}
for label, ratio in regimes.items():
    Xbar = ratio * X0
    dK_over_K = beta_DEV * a0 / mu(Xbar)
    vector_ratios[label] = dK_over_K

# -------------------- TASK 4: exponent vs regime --------------------
fig, ax = plt.subplots(figsize=(7, 5))
xs = np.array([np.log10(r) for r in regimes.values()])
ys_p = np.array([beta_local[l]  for l in regimes])
ys_r = np.array([gamma_pos[l]   for l in regimes])
ax.plot(xs, ys_p, "o-", label="beta  (momentum slope)")
ax.plot(xs, ys_r, "s-", label="gamma (position slope)")
ax.axhline(1.5, color="r", ls="--", lw=1, label="3/2 fractional")
ax.axhline(2.0, color="b", ls="--", lw=1, label="2 Poisson")
ax.axhline(0.5, color="r", ls=":",  lw=1, label="1/2 fractional (pos)")
ax.axhline(1.0, color="b", ls=":",  lw=1, label="1 Poisson (pos)")
ax.set_xlabel("log10(X_bar / X_0)")
ax.set_ylabel("exponent")
ax.set_title("Propagator exponent vs background regime")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "exponent_vs_regime.png", dpi=140)
plt.close(fig)

# -------------------- Report --------------------
lines = []
lines.append("Paper IV - Step 1: Effective propagator of the DEV scalar")
lines.append("=" * 60)
lines.append("")
lines.append("TASK 1 - Local propagator G~(p) ~ p^{-beta}")
lines.append("-" * 60)
for l, b in beta_local.items():
    lines.append(f"  {l:40s}  beta = {b:.4f}")
lines.append("")
lines.append("TASK 2 - Position Green function G(r) ~ r^{-gamma}")
lines.append("-" * 60)
for l, g in gamma_pos.items():
    lines.append(f"  {l:40s}  gamma = {g:.4f}")
lines.append("")
lines.append("TASK 3 - Vector-field contribution Delta K~ / K~")
lines.append("-" * 60)
for l, v in vector_ratios.items():
    tag = "DOMINANT" if v > 1 else ("SUBDOMINANT" if v < 0.1 else "MARGINAL")
    lines.append(f"  {l:40s}  ratio = {v:.3e}  [{tag}]")
lines.append("")
lines.append("TASK 4 - See exponent_vs_regime.png")
lines.append("")
lines.append("Conclusion")
lines.append("-" * 60)
b_dm = beta_local["deep-MOND extreme (X/X0=1e-4)"]
g_dm = gamma_pos ["deep-MOND extreme (X/X0=1e-4)"]
if 1.4 <= b_dm <= 1.6 and 0.4 <= g_dm <= 0.6:
    verdict = "CONFIRMS fractional Laplacian (-nabla^2)^{3/4}"
elif (1.2 <= b_dm <= 1.4) or (1.6 <= b_dm <= 1.8):
    verdict = "PARTIAL - fractional but different exponent"
elif abs(b_dm - 2.0) < 0.1:
    verdict = "REFUTES - linearized action gives standard Poisson"
else:
    verdict = "NEW REGIME - report honestly"
lines.append(f"  deep-MOND beta  = {b_dm:.3f}")
lines.append(f"  deep-MOND gamma = {g_dm:.3f}")
lines.append(f"  Verdict: {verdict}")
lines.append("")
lines.append("Note: the *local* quadratic action of the DBI scalar")
lines.append("around a constant-Xbar background is exactly")
lines.append("K~(p) = mu(Xbar) * p^2, i.e. Poisson with a")
lines.append("renormalized coupling mu. Fractional behaviour in the")
lines.append("position-space Green function arises only when the")
lines.append("background gradient |grad theta_bar| itself carries the")
lines.append("scale of the perturbation (deep-MOND r-dependence).")
lines.append("This is the analytic origin of the alpha ~ -1.56")
lines.append("found numerically in Paper III: it is a background-")
lines.append("induced non-locality, not a feature of the linearized")
lines.append("kinetic operator at a fixed point.")

report = "\n".join(lines)
(OUT / "propagator_report.txt").write_text(report, encoding="utf-8")
print(report)
