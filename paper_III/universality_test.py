"""
Universality test for alpha = -1.56 in the DEV non-local operator.

For each of 4 mass profiles, compute S_eff(r) = nabla^2 f(r) and
fit a power law S_eff ~ r^alpha in the deep-MOND window.
"""
import numpy as np
import matplotlib.pyplot as plt

# Constants (SI)
a0   = 1.2e-10            # m/s^2
beta = 0.0075
G    = 6.674e-11          # m^3 kg^-1 s^-2
KPC  = 3.086e19           # m
MSUN = 1.989e30           # kg

def nu(y):
    return np.sqrt(0.5 + 0.5*np.sqrt(1.0 + 4.0/y**2))

def plummer_M(r, M, a):
    """Plummer enclosed mass."""
    return M * r**3 / (r**2 + a**2)**1.5

def hernquist_M(r, M, rs):
    return M * r**2 / (r + rs)**2

def compute_alpha(name, Mfunc, Mtot_kg, r_scale_m, label):
    # log-spaced grid in meters
    r = np.logspace(np.log10(1e-2*KPC), np.log10(1e4*KPC), 4000)
    Mr = Mfunc(r)

    gN   = G * Mr / r**2
    gobs = nu(gN/a0) * gN

    # Psi(r) = -int_r^infty gobs dr'  (we use |Psi|, sign cancels later)
    # integrate from outside in
    Psi = np.zeros_like(r)
    # trapezoidal cumulative from the right
    for i in range(len(r)-2, -1, -1):
        Psi[i] = Psi[i+1] + 0.5*(gobs[i] + gobs[i+1])*(r[i+1]-r[i])
    # Psi here is positive magnitude of the integral

    # eta_anal - 1
    x = gN/a0
    eta_m1 = (2.0*beta/3.0) / np.sqrt(x*(1.0+x))

    f = eta_m1 * Psi  # sign of Psi absorbed; we want magnitude scaling

    # S_eff = (1/r^2) d/dr [ r^2 df/dr ]
    df_dr = np.gradient(f, r)
    inner = r**2 * df_dr
    Seff  = np.gradient(inner, r) / r**2

    # deep-MOND window: r > 3 r_MOND, r < 0.3 R_cut
    rMOND = np.sqrt(G*Mtot_kg/a0)
    Rcut  = r[-1]
    mask = (r > 3.0*rMOND) & (r < 0.3*Rcut) & (np.abs(Seff) > 0) & np.isfinite(Seff)

    rr = r[mask]
    ss = np.abs(Seff[mask])
    # power-law fit
    coeff = np.polyfit(np.log(rr), np.log(ss), 1)
    alpha = coeff[0]
    # residual std as uncertainty
    resid = np.log(ss) - (coeff[0]*np.log(rr) + coeff[1])
    sigma = np.std(resid) / np.sqrt(len(rr))
    # convert sigma on intercept to slope uncertainty (rough)
    n = len(rr)
    x_ = np.log(rr)
    slope_err = np.std(resid) / np.sqrt(np.sum((x_-x_.mean())**2))

    return dict(name=name, label=label, r=r, Seff=Seff, mask=mask,
                alpha=alpha, alpha_err=slope_err, rMOND=rMOND,
                fit_intercept=coeff[1])

# Profiles
profiles = []

# A: Plummer point-source reference
profiles.append(compute_alpha(
    "A_pointsource",
    lambda r: plummer_M(r, 1e10*MSUN, 0.01*KPC),
    1e10*MSUN, 0.01*KPC,
    r"A: Plummer point ($\varepsilon\!\approx\!0$)"
))

# B: NGC1052-DF2
profiles.append(compute_alpha(
    "B_DF2",
    lambda r: plummer_M(r, 2e8*MSUN, 2.2*KPC),
    2e8*MSUN, 2.2*KPC,
    r"B: DF2 ($\varepsilon=4.56$)"
))

# C: DGSAT-I
profiles.append(compute_alpha(
    "C_DGSAT",
    lambda r: plummer_M(r, 3e8*MSUN, 4.7*KPC),
    3e8*MSUN, 4.7*KPC,
    r"C: DGSAT-I ($\varepsilon=7.96$)"
))

# D: Hernquist massive spiral
profiles.append(compute_alpha(
    "D_Hernquist",
    lambda r: hernquist_M(r, 5e11*MSUN, 10*KPC),
    5e11*MSUN, 10*KPC,
    r"D: Hernquist ($\varepsilon\!\gg\!1$)"
))

# Epsilons (for Fig. 2)
eps = [0.0, 4.56, 7.96, 20.0]

# Figure 1: 4 panels
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, p in zip(axes.flat, profiles):
    r = p['r']/KPC
    ax.loglog(r, np.abs(p['Seff']), 'k-', lw=1.2, label=r'$|S_{\rm eff}|$')
    rr = p['r'][p['mask']]/KPC
    fit = np.exp(p['fit_intercept']) * (p['r'][p['mask']])**p['alpha']
    ax.loglog(rr, fit, 'r--', lw=1.5,
              label=fr"fit $\alpha={p['alpha']:.3f}\pm{p['alpha_err']:.3f}$")
    ax.axvline(p['rMOND']/KPC, color='gray', ls=':', alpha=0.6, label=r'$r_{\rm MOND}$')
    ax.set_xlabel(r'$r$ [kpc]')
    ax.set_ylabel(r'$|S_{\rm eff}|$')
    ax.set_title(p['label'])
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('paper_III/universality_Seff.png', dpi=140)
plt.close()

# Figure 2: alpha vs epsilon
fig, ax = plt.subplots(figsize=(7, 5))
alphas = [p['alpha'] for p in profiles]
errs   = [p['alpha_err'] for p in profiles]
ax.errorbar(eps, alphas, yerr=errs, fmt='o', ms=8, capsize=4, color='navy')
ax.axhline(-2.0,  color='black', ls='--', label=r'Poisson $\alpha=-2$')
ax.axhline(-1.56, color='red',   ls=':',  label=r'deep-MOND point $\alpha=-1.56$')
for p, e in zip(profiles, eps):
    ax.annotate(p['name'].split('_')[0], xy=(e, p['alpha']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax.set_xlabel(r'$\varepsilon$')
ax.set_ylabel(r'$\alpha$')
ax.set_title(r'Universality of $\alpha$ across mass profiles')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('paper_III/universality_alpha_vs_epsilon.png', dpi=140)
plt.close()

# Report
alpha_A = profiles[0]['alpha']
diffs   = [abs(p['alpha'] - alpha_A) for p in profiles[1:]]
maxdiff = max(diffs)
if maxdiff < 0.05:
    verdict = "UNIVERSAL"
elif maxdiff < 0.10:
    verdict = "MARGINAL"
else:
    verdict = "NOT UNIVERSAL"

with open('paper_III/universality_report.txt', 'w', encoding='utf-8') as fh:
    fh.write("="*64 + "\n")
    fh.write("UNIVERSALITY TEST FOR alpha (DEV non-local operator)\n")
    fh.write("="*64 + "\n\n")
    fh.write(f"{'Profile':<22}{'epsilon':>10}{'alpha':>12}{'sigma':>12}\n")
    fh.write("-"*56 + "\n")
    for p, e in zip(profiles, eps):
        fh.write(f"{p['name']:<22}{e:>10.2f}{p['alpha']:>12.4f}{p['alpha_err']:>12.4f}\n")
    fh.write("\n")
    fh.write(f"|alpha_A - alpha_B| = {abs(profiles[1]['alpha']-alpha_A):.4f}\n")
    fh.write(f"|alpha_A - alpha_C| = {abs(profiles[2]['alpha']-alpha_A):.4f}\n")
    fh.write(f"|alpha_A - alpha_D| = {abs(profiles[3]['alpha']-alpha_A):.4f}\n")
    fh.write(f"max |Delta alpha|  = {maxdiff:.4f}\n\n")
    fh.write(f"VERDICT: {verdict}\n\n")
    if verdict == "UNIVERSAL":
        fh.write("alpha is universal across mass profiles spanning\n")
        fh.write("epsilon in [0, ~20] and M in [2e8, 5e11] Msun.\n")
        fh.write("Recommendation: include in Paper III Section II.\n")
    elif verdict == "MARGINAL":
        fh.write("alpha varies marginally; report with caution.\n")
    else:
        fh.write("alpha depends on the profile. Report honestly\n")
        fh.write("as a limitation; first-principles derivation is\n")
        fh.write("the primary objective of Paper IV.\n")

print(f"VERDICT: {verdict}, max Delta alpha = {maxdiff:.4f}")
for p, e in zip(profiles, eps):
    print(f"  {p['name']:<22} eps={e:>6.2f}  alpha = {p['alpha']:+.4f} +/- {p['alpha_err']:.4f}")
