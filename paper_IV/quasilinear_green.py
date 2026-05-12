"""
Paper IV - Step 2: Green function of the quasilinear operator
  L = nabla . [ mu(|grad theta_bar|/a0) grad d-theta ]
in the deep-MOND background, for 4 mass profiles.

Units: kpc, km/s, Msun.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------- canonical constants --------------------
a0 = 3703.0                        # (km/s)^2 / kpc
G  = 4.302e-3                      # kpc (km/s)^2 / Msun

OUT = Path(__file__).parent

# -------------------- DEV interpolating functions --------------
def mu_fn(x):
    return x / np.sqrt(1.0 + x*x)

def nu_fn(y):
    # y = g_N / a0
    return np.sqrt(0.5 + 0.5*np.sqrt(1.0 + 4.0/(y*y)))

# -------------------- mass profiles --------------------
def M_point(r, M):           return np.full_like(r, M)
def M_plummer(r, M, re):     return M * r**3 / (r*r + re*re)**1.5
def M_hernquist(r, M, rs):   return M * r*r / (r + rs)**2

profiles = {
    "A: point  M=1e10":           ("point",     dict(M=1e10)),
    "B: NGC1052-DF2 (Plummer)":   ("plummer",   dict(M=2e8,  re=2.2)),
    "C: DGSAT-I  (Plummer)":      ("plummer",   dict(M=3e8,  re=4.7)),
    "D: Hernquist M=5e11":        ("hernquist", dict(M=5e11, rs=10.0)),
}

def Menc(r, kind, pars):
    if kind == "point":     return M_point(r, **pars)
    if kind == "plummer":   return M_plummer(r, **pars)
    if kind == "hernquist": return M_hernquist(r, **pars)
    raise ValueError(kind)

def r_MOND(kind, pars):
    # characteristic MOND radius based on the total/asymptotic mass
    Mtot = pars["M"]
    return np.sqrt(G*Mtot/a0)

# -------------------- coefficient w(r) --------------------
def w_of_r(r, kind, pars):
    Me = Menc(r, kind, pars)
    gN = G * Me / (r*r)
    y  = gN / a0
    nu = nu_fn(y)
    gobs = nu * gN
    return mu_fn(gobs / a0)

# -------------------- Green function via quadrature --------------
# spherical symmetry, source at origin:
#   (1/r^2) d/dr [ r^2 w(r) dG/dr ] = -delta^3(r)  (sign: attractive)
# Integrating a sphere: 4 pi r^2 w(r) dG/dr = -1
#   dG/dr = -1 / (4 pi r^2 w(r))
# G(r) = int_r^{Rcut} ds / (4 pi s^2 w(s))   so that G(Rcut)=0.
def green(r, kind, pars):
    # use a fine logarithmic grid for the inner integral, then sample at r
    s = np.logspace(np.log10(r[0]) - 0.5, np.log10(r[-1]) + 0.5, 8000)
    integrand = 1.0 / (4.0*np.pi * s*s * w_of_r(s, kind, pars))
    # cumulative integral from the right
    seg = 0.5*(integrand[:-1] + integrand[1:]) * np.diff(s)
    cum = np.concatenate(([0.0], np.cumsum(seg)))    # int_{s0}^{s_i}
    Gfull = cum[-1] - cum    # int_{s_i}^{s_end}
    return np.interp(r, s, Gfull)

# -------------------- fit slope in deep-MOND window --------------
def fit_slope(r, G_arr, rlo, rhi):
    m = (r >= rlo) & (r <= rhi) & (G_arr > 0)
    if m.sum() < 5:
        return np.nan, np.nan
    lr = np.log(r[m]); lg = np.log(G_arr[m])
    p, cov = np.polyfit(lr, lg, 1, cov=True)
    return -p[0], np.sqrt(cov[0,0])     # gamma, sigma

# -------------------- main loop --------------------
r = np.logspace(-2, 4, 4000)
results = {}

fig1, ax1 = plt.subplots(figsize=(7.5, 5.5))
colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(profiles)))

for (label, (kind, pars)), col in zip(profiles.items(), colors):
    Garr = green(r, kind, pars)
    rM = r_MOND(kind, pars)
    rlo, rhi = 3.0*rM, 0.3*r[-1]
    gamma, sigma = fit_slope(r, Garr, rlo, rhi)
    alpha_pred = -(1.0 + 2.0*gamma)
    results[label] = dict(gamma=gamma, sigma=sigma, alpha=alpha_pred, rM=rM)
    ax1.loglog(r, np.abs(Garr), color=col,
               label=f"{label}: gamma={gamma:.3f}+/-{sigma:.3f}")
    ax1.axvline(rM, color=col, ls=":", lw=0.7, alpha=0.6)

# reference slopes (normalize at r=10 kpc for readability)
ref_norm = 1e-2
r_ref = r[(r > 1) & (r < 100)]
ax1.loglog(r_ref, ref_norm * (r_ref/10.0)**-0.28, "r--", lw=1.0,
           label="r^-0.28 (Paper III alpha=-1.56)")
ax1.loglog(r_ref, ref_norm * (r_ref/10.0)**-0.5,  "b--", lw=1.0,
           label="r^-0.5 (fractional)")
ax1.loglog(r_ref, ref_norm * (r_ref/10.0)**-1.0,  color="gray", ls="--", lw=1.0,
           label="r^-1 (Poisson)")
ax1.set_xlabel("r [kpc]")
ax1.set_ylabel("|G(r)|")
ax1.set_title("Quasilinear Green function L = div[mu(|grad theta|/a0) grad]")
ax1.legend(fontsize=8, loc="lower left")
ax1.grid(True, which="both", alpha=0.3)
fig1.tight_layout()
fig1.savefig(OUT / "quasilinear_green_function.png", dpi=140)
plt.close(fig1)

# Figure 2: gamma vs profile
fig2, ax2 = plt.subplots(figsize=(7.0, 4.5))
xs = np.arange(len(profiles))
ys = [results[l]["gamma"] for l in profiles]
es = [results[l]["sigma"] for l in profiles]
ax2.errorbar(xs, ys, yerr=es, fmt="o", color="k", capsize=4)
ax2.axhline(0.28, color="r", ls="--", lw=1, label="gamma=0.28 (Paper III target)")
ax2.axhline(0.5,  color="b", ls="--", lw=1, label="gamma=0.5 (fractional)")
ax2.axhline(1.0,  color="gray", ls="--", lw=1, label="gamma=1.0 (Poisson)")
ax2.set_xticks(xs); ax2.set_xticklabels(list(profiles.keys()), rotation=20, ha="right", fontsize=8)
ax2.set_ylabel("gamma (Green-function exponent)")
ax2.set_title("Quasilinear Green exponent across mass profiles")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(OUT / "gamma_vs_profile.png", dpi=140)
plt.close(fig2)

# -------------------- report --------------------
alpha_III = {
    "A: point  M=1e10":         -1.56,
    "B: NGC1052-DF2 (Plummer)": -1.45,
    "C: DGSAT-I  (Plummer)":    -1.41,
    "D: Hernquist M=5e11":      -1.96,
}

lines = []
lines.append("Paper IV - Step 2: Quasilinear Green function")
lines.append("=" * 64)
lines.append("")
lines.append(f"{'Profile':32s} {'gamma':>10s} {'sigma':>8s} "
             f"{'alpha_pred':>12s} {'alpha_III':>10s} {'agree?':>8s}")
lines.append("-" * 84)
ok = 0; total = 0
for l, d in results.items():
    a_pred = d["alpha"]
    a_III  = alpha_III.get(l, np.nan)
    agree  = abs(a_pred - a_III)/abs(a_III) < 0.10
    ok    += int(agree); total += 1
    lines.append(f"{l:32s} {d['gamma']:>10.4f} {d['sigma']:>8.4f} "
                 f"{a_pred:>12.4f} {a_III:>10.4f} {('YES' if agree else 'NO'):>8s}")
lines.append("")
lines.append(f"r_MOND values (kpc):")
for l, d in results.items():
    lines.append(f"  {l:32s} r_MOND = {d['rM']:.3f}")
lines.append("")
g_udg = 0.5*(results["B: NGC1052-DF2 (Plummer)"]["gamma"]
            + results["C: DGSAT-I  (Plummer)"]["gamma"])
a_udg_pred = -(1.0 + 2.0*g_udg)
lines.append(f"UDG mean: gamma = {g_udg:.3f}, alpha_pred = {a_udg_pred:.3f}")
lines.append("")
if ok == total:
    verdict = "CONFIRMS - quasilinear operator reproduces Paper III alpha"
elif ok >= total // 2:
    verdict = "PARTIAL - mechanism correct, residual mismatch"
else:
    verdict = "REFUTES - additional mechanism required"
lines.append(f"Verdict: {verdict}  ({ok}/{total} profiles agree to 10%)")
lines.append("")
lines.append("Relation tested: alpha_Paper_III = -(1 + 2 * gamma_Green).")
lines.append("If verified, the non-local exponent of Paper III is the")
lines.append("direct signature of the Bekenstein-Milgrom-type quasilinear")
lines.append("operator L = div[mu(|grad theta_bar|/a0) grad] acting on the")
lines.append("perturbation in the deep-MOND background of the DEV action.")

report = "\n".join(lines)
(OUT / "quasilinear_report.txt").write_text(report, encoding="utf-8")
print(report)
