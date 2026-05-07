"""
beta_naturalness.py
Numerical/dimensional exploration of the origin of beta = gamma^2/(m_A^2 a0)
in DEV theory. Calibrated empirically as beta_best = 0.0075.

Goal: identify (or honestly rule out) a natural dimensional origin.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Constants (SI) ----------------
H0 = 2.268e-18           # s^-1
rho_L = 6.0e-33          # kg/m^3
Omega_L = 0.70
Omega_m = 0.30
c = 3.0e8                # m/s
G = 6.674e-11            # m^3 kg^-1 s^-2
a0 = 1.2e-10             # m/s^2  (MOND scale)
alpha_fs = 1/137.036
e = 1.602e-19            # C
eV = 1.602e-19           # J

rho_crit_recompute = 3*H0**2 / (8*np.pi*G)
rho_crit = 9.47e-27      # given
X0 = a0**2 / 2.0
rho_vac_DEV = X0/(8*np.pi*G)

beta_best = 0.0075

lines = []
def P(s=""):
    print(s)
    lines.append(s)

P("="*78)
P("BETA NATURALNESS — DEV theory")
P("="*78)
P(f"H0={H0:.3e} s^-1   rho_L={rho_L:.3e} kg/m^3   rho_crit(given)={rho_crit:.3e}")
P(f"rho_crit(recomputed 3H0^2/8piG) = {rho_crit_recompute:.3e}  kg/m^3")
P(f"a0={a0:.3e}   X0=a0^2/2 = {X0:.3e}   rho_vac_DEV=X0/(8piG)={rho_vac_DEV:.3e} kg/m^3")
P(f"beta_best (empirical) = {beta_best}")
P("")

# ---------------- Task 1: dimensional sweep ----------------
candidates = {
    "A  sqrt(rho_L/rho_crit)"               : np.sqrt(rho_L/rho_crit),
    "B  rho_L/rho_vac_DEV"                  : rho_L/rho_vac_DEV,
    "C  sqrt(rho_L/rho_vac_DEV)"            : np.sqrt(rho_L/rho_vac_DEV),
    "D  (rho_L/rho_crit)^(1/3)"             : (rho_L/rho_crit)**(1/3),
    "E  a0/(c*H0)"                          : a0/(c*H0),
    "F  sqrt(a0/(c*H0))"                    : np.sqrt(a0/(c*H0)),
    "G  (a0/(c*H0))^(1/3)"                  : (a0/(c*H0))**(1/3),
    "H  (a0/(c*H0))^2 / Omega_L"            : (a0/(c*H0))**2 / Omega_L,
    "I  sqrt(rho_L*G/a0^2)/c"               : np.sqrt(rho_L*G/a0**2)/c,
    "J  sqrt(G*rho_L)/a0"                   : np.sqrt(G*rho_L)/a0,
    "K  sqrt(G*H0/a0)/c"                    : np.sqrt(G*H0/a0)/c,
    "L  (2/3)*sqrt(rho_L/rho_crit)"         : (2/3)*np.sqrt(rho_L/rho_crit),
    "M  (1/(4pi))*sqrt(rho_L/rho_crit)"     : (1/(4*np.pi))*np.sqrt(rho_L/rho_crit),
    "N  3*sqrt(rho_L/rho_crit)"             : 3*np.sqrt(rho_L/rho_crit),
    "O  sqrt(3*rho_L/rho_crit)"             : np.sqrt(3*rho_L/rho_crit),
}

results = []
for name, val in candidates.items():
    if val <= 0 or not np.isfinite(val):
        continue
    ratio = val/beta_best
    logr = abs(np.log10(ratio))
    if logr < 0.3:   cat = "VERY INTERESTING"
    elif logr < 0.7: cat = "INTERESTING"
    elif logr <= 1.0:cat = "MARGINAL"
    else:            cat = "DISCARDED"
    results.append((name, val, ratio, logr, cat))

results.sort(key=lambda r: r[3])

P("-"*78)
P("TASK 1 — Dimensional sweep (sorted by |log10(ratio)|)")
P("-"*78)
P(f"{'Candidate':40s} {'value':>12s} {'ratio':>10s} {'|log10|':>9s}  category")
for name,val,ratio,logr,cat in results:
    P(f"{name:40s} {val:12.4e} {ratio:10.3f} {logr:9.3f}  {cat}")
P("")

with open("beta_dimensional_table.txt","w") as f:
    f.write(f"{'Candidate':40s} {'value':>12s} {'ratio':>10s} {'|log10|':>9s}  category\n")
    for name,val,ratio,logr,cat in results:
        f.write(f"{name:40s} {val:12.4e} {ratio:10.3f} {logr:9.3f}  {cat}\n")

# ---------------- Task 2: factor ~9 ----------------
P("-"*78)
P("TASK 2 — Factor ~9 search")
P("-"*78)
betaA = np.sqrt(rho_L/rho_crit)
target_factor = beta_best / betaA
P(f"beta_A = sqrt(rho_L/rho_crit) = {betaA:.4e}")
P(f"target factor beta_best/beta_A = {target_factor:.4f}")

factor_candidates = {
    "3^2 = 9"                : 9.0,
    "4pi/sqrt(4pi/3)"        : 4*np.pi/np.sqrt(4*np.pi/3),
    "1/(2*alpha_fs)"         : 1/(2*alpha_fs),
    "Omega_m/Omega_L^2"      : Omega_m/Omega_L**2,
    "3*Omega_m/2"            : 3*Omega_m/2,
    "15"                     : 15.0,
}
for n,v in factor_candidates.items():
    P(f"  {n:25s} = {v:10.4f}    diff_from_target = {abs(v-target_factor):.4f}")

P("")
P("Sweep N_DOF^2 * sqrt(rho_L/rho_crit), N=1..5:")
best = None
for N in range(1,6):
    val = N**2 * betaA
    ratio = val/beta_best
    P(f"  N={N}   N^2*beta_A = {val:.4e}    ratio={ratio:.3f}")
    if best is None or abs(np.log10(ratio)) < abs(np.log10(best[1]/beta_best)):
        best = (N, val)
P(f"Best N: N={best[0]}  value={best[1]:.4e}  ratio={best[1]/beta_best:.3f}")
P("")

# ---------------- Task 3: scale consistency ----------------
P("-"*78)
P("TASK 3 — Scale consistency from physical systems")
P("-"*78)
systems = [
    (1.0,   1.005, 0.040),
    (0.5,   0.990, 0.050),
    (0.010, 1.040, 0.070),
    (0.005, 1.080, 0.100),
    (0.003, 1.100, 0.120),
]
g_over_a0 = []
beta_vals = []
beta_errs = []
for g_a, eta, sig in systems:
    y = g_a
    factor = np.sqrt(y*(1+y))
    # beta_implied = (eta-1) * sqrt(y*(1+y)) / (2/3)
    beta_imp = (eta-1.0) * factor / (2/3)
    sig_beta = sig * factor / (2/3)
    g_over_a0.append(g_a)
    beta_vals.append(beta_imp)
    beta_errs.append(sig_beta)
    P(f"  g/a0={g_a:7.3f}  eta={eta:.3f} +/- {sig:.3f}    beta_imp={beta_imp:.4f} +/- {sig_beta:.4f}")

g_over_a0 = np.array(g_over_a0)
beta_vals = np.array(beta_vals)
beta_errs = np.array(beta_errs)

# Linear fit beta vs log10(g/a0), weighted
x = np.log10(g_over_a0)
w = 1.0/beta_errs**2
S   = np.sum(w)
Sx  = np.sum(w*x)
Sy  = np.sum(w*beta_vals)
Sxx = np.sum(w*x*x)
Sxy = np.sum(w*x*beta_vals)
D   = S*Sxx - Sx*Sx
slope = (S*Sxy - Sx*Sy)/D
intercept = (Sxx*Sy - Sx*Sxy)/D
sig_slope = np.sqrt(S/D)
sig_int   = np.sqrt(Sxx/D)

P("")
P(f"Weighted linear fit: beta = a + b*log10(g/a0)")
P(f"  intercept a = {intercept:.4f} +/- {sig_int:.4f}")
P(f"  slope     b = {slope:.4f} +/- {sig_slope:.4f}")
n_sig = abs(slope)/sig_slope
P(f"  slope is {n_sig:.2f}-sigma from zero")
if n_sig < 2:
    verdict_scale = f"slope consistent with 0 within 2 sigma -> beta is scale-invariant"
else:
    verdict_scale = f"slope NOT consistent with 0 within 2 sigma -> possible scale dependence"
P(f"  VERDICT: {verdict_scale}")
P(f"  beta range across systems: [{beta_vals.min():.4f}, {beta_vals.max():.4f}]")

# Plot
fig, ax = plt.subplots(figsize=(7,5))
ax.errorbar(g_over_a0, beta_vals, yerr=beta_errs, fmt='o', capsize=4,
            color='C0', label='implied beta')
ax.axhline(beta_best, color='red', linestyle='--', label=f'beta_best = {beta_best}')
xx = np.logspace(np.log10(g_over_a0.min()*0.5), np.log10(g_over_a0.max()*2), 200)
ax.plot(xx, intercept + slope*np.log10(xx), 'k:', label='weighted fit')
ax.set_xscale('log')
ax.set_xlabel(r'$g/a_0$')
ax.set_ylabel(r'$\beta_{\rm implied}$')
ax.set_title('Scale consistency of beta')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('beta_scale_consistency.png', dpi=140)
plt.close()
P(f"  -> beta_scale_consistency.png written")
P("")

# ---------------- Task 4: coincidence pairs ----------------
P("-"*78)
P("TASK 4 — Coincidence pairs C1^alpha * C2^gamma")
P("-"*78)
C1 = a0/(c*H0)
C2 = np.sqrt(rho_L/rho_crit)
P(f"C1 = a0/(c*H0)        = {C1:.4e}")
P(f"C2 = sqrt(rho_L/rho_c)= {C2:.4e}")
exps = [-2, -1, -0.5, 0.5, 1, 2]
hits = []
for a_ in exps:
    for g_ in exps:
        v = (C1**a_) * (C2**g_)
        r = v/beta_best
        if 0.5 <= r <= 2.0:
            hits.append((a_, g_, v, r))
hits.sort(key=lambda h: abs(np.log10(h[3])))
if hits:
    P("Pairs within factor 2 of beta_best:")
    for a_, g_, v, r in hits:
        P(f"  C1^{a_:+.1f} * C2^{g_:+.1f} = {v:.4e}   ratio={r:.3f}")
    best_pair = hits[0]
else:
    P("No (alpha,gamma) pair within factor 2.")
    best_pair = None
P("")

# ---------------- Task 5: first-principles ----------------
P("-"*78)
P("TASK 5 — First-principles estimates of gamma")
P("-"*78)
mA_eV = 3.7e-25                  # eV/c^2 lower bound
mA_kg = mA_eV * eV / c**2
P(f"m_A (lower bound) = {mA_eV:.2e} eV/c^2 = {mA_kg:.3e} kg")

# beta = gamma^2/(m_A^2 a0)  -- treating in SI (whatever dimensions of gamma)
gamma_SI_sq = beta_best * mA_kg**2 * a0
gamma_SI    = np.sqrt(gamma_SI_sq)
P(f"gamma (SI, from beta=gamma^2/(m_A^2 a0)) = {gamma_SI:.3e}   (units: kg*sqrt(m/s^2))")

# Natural-units conversion: hbar = c = 1
# m_A in eV; a0 in eV (1/length): a0 has units of acceleration -> in natural units it's energy^2
hbar = 1.054571817e-34
# a0 [m/s^2] -> in natural units (energy units): a0_nat[eV] = hbar*a0/c / eV  (length^-1 -> energy)
# Cleaner: gamma in DEV is a coupling; assume gamma dimensionless in natural units.
# Compute gamma_natural = sqrt(beta) * m_A * sqrt(a0_nat)/m_A   ~ depends on convention.
# Provide an estimate assuming gamma is dimensionless:
gamma_nat = np.sqrt(beta_best)
P(f"If gamma is dimensionless in natural units: gamma ~ sqrt(beta) = {gamma_nat:.4f}")

# Dark-photon mixing gamma ~ epsilon * e
eps_required = gamma_nat / 1.0   # treating gamma_nat as the mixing-strength estimator (dimensionless)
P(f"Dark-photon kinetic-mixing interpretation: gamma ~ epsilon*e")
eps_from_e = gamma_nat / e if False else gamma_nat  # we keep dimensionless
P(f"  Required epsilon (dimensionless) ~ sqrt(beta) = {gamma_nat:.4f} ~ 9e-2")
P(f"  Astrophysical/lab bound on epsilon ~ 1e-3")
P(f"  -> Required epsilon EXCEEDS bound by ~ {gamma_nat/1e-3:.1f}x")
P(f"     => DEV gamma is NOT a generic dark-photon kinetic mixing.")
P("")

# ---------------- Final summary ----------------
P("="*78)
P("SUMMARY")
P("="*78)
top3 = results[:3]
P("Top 3 dimensional matches:")
for name,val,ratio,logr,cat in top3:
    P(f"  {name}  value={val:.4e}  ratio={ratio:.3f}  [{cat}]")
P("")
P(f"Factor-9 best: N=3 -> 9*sqrt(rho_L/rho_crit) = {9*betaA:.4e}  (ratio={9*betaA/beta_best:.3f})")
P(f"Scale slope: {slope:.4f} +/- {sig_slope:.4f}  ({n_sig:.2f} sigma); {verdict_scale}")
if best_pair:
    P(f"Best (alpha,gamma): C1^{best_pair[0]} * C2^{best_pair[1]} -> ratio {best_pair[3]:.3f}")
else:
    P("Best (alpha,gamma): none within factor 2.")
P(f"Required epsilon ~ {gamma_nat:.3f}, bound 1e-3 -> exceeds by ~{gamma_nat/1e-3:.0f}x")

with open("beta_naturalness_report.txt","w") as f:
    f.write("\n".join(lines))

print("\n[wrote] beta_dimensional_table.txt, beta_scale_consistency.png, beta_naturalness_report.txt")
