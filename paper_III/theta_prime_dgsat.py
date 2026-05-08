"""
Paper III - Direct integration of the DEV scalar field equation
for the DGSAT-I Plummer profile.

Equation (spherical, quasi-static):
   d/dr [ r^2 mu(|grad Phi|/a0) theta'(r) ] = 4 pi G rho_b(r) r^2
=> r^2 mu(g/a0) theta'(r) = G M(r)
=> theta'(r) = G M(r) / [ r^2 mu(g/a0) ]

The local field strength g obeys the AQUAL equation:
   mu(g/a0) g = g_N(r) = G M(r)/r^2
solved pointwise in r.

We compare two quantities (both squared):
  A_form (deep-MOND naive):   (theta')_A^2 = g_N^2 / a0^2
  B_form (exact spherical):   (theta')_B^2 = [ g_N / (mu*a0) ]^2

In SI throughout.  At the end we plot ratio (theta')_A / (theta')_B
and the two profiles in dimensionless form.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# SI constants
G   = 6.6743e-11        # m^3 kg^-1 s^-2
a0  = 1.2e-10           # m/s^2
Msun = 1.98892e30       # kg
kpc  = 3.0857e19        # m

# DGSAT-I Plummer (Paper II Table I)
M_total = 3.0e8 * Msun
r_eff   = 2.1 * kpc

def Menc(r):
    return M_total * r**3 / (r**2 + r_eff**2)**1.5

def gN(r):
    return G * Menc(r) / r**2

def nu(y):
    """MOND interpolating function nu(y) = sqrt(0.5+0.5*sqrt(1+4/y^2))."""
    return np.sqrt(0.5 + 0.5*np.sqrt(1.0 + 4.0/y**2))

def mu_of_x(x):
    """MOND mu(x) consistent with above nu, where x = g/a0."""
    return x / np.sqrt(1.0 + x**2)   # standard interpolation

def solve_g_from_gN(gN_val):
    """Solve mu(g/a0)*g = gN for g.  With mu = x/sqrt(1+x^2),
       this becomes g^2/sqrt(a0^2+g^2) = gN."""
    f = lambda g: g*g/np.sqrt(a0*a0 + g*g) - gN_val
    g_lo = 1e-6 * gN_val
    g_hi = max(gN_val, np.sqrt(gN_val*a0)) * 10.0
    while f(g_hi) < 0:
        g_hi *= 10
    return brentq(f, g_lo, g_hi)

# r grid
N = 400
r = np.geomspace(0.05*r_eff, 30.0*r_eff, N)
r_kpc = r / kpc

g_N_arr = gN(r)
g_obs   = np.array([solve_g_from_gN(gNv) for gNv in g_N_arr])
mu_arr  = mu_of_x(g_obs/a0)

# r_MOND
r_MOND = np.sqrt(G*M_total/a0)
print(f"r_MOND for DGSAT-I = {r_MOND/kpc:.3f} kpc")
print(f"r_eff             = {r_eff/kpc:.3f} kpc")
print(f"epsilon = r_eff/r_MOND = {r_eff/r_MOND:.3f}")

# Two forms
thetaA2 = (g_N_arr / a0)**2                    # naive deep-MOND
thetaB2 = (g_N_arr / (mu_arr * a0))**2         # exact spherical AQUAL

ratio = thetaA2 / thetaB2     # = mu^2  -- by construction
y     = g_N_arr / a0          # dimensionless Newtonian field

# When does naive A agree with exact B (within 10%)?
diff = np.abs(ratio - 1.0)
agree_mask = diff < 0.10
worst_diff_pct = diff.max() * 100
median_diff_pct = np.median(diff) * 100

# Identify regimes
i_eff = int(np.argmin(np.abs(r - r_eff)))
i_3eff = int(np.argmin(np.abs(r - 3*r_eff)))
i_inner = int(np.argmin(np.abs(r - 0.3*r_eff)))

# ---- plot ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].loglog(r_kpc, np.sqrt(thetaA2), label=r"$|\theta'|_A = g_N/a_0$ (naive DM)")
ax[0].loglog(r_kpc, np.sqrt(thetaB2), '--', label=r"$|\theta'|_B = g_N/(\mu a_0)$ (exact)")
ax[0].axvline(r_eff/kpc, color='gray', ls=':', label=r"$r_{\rm eff}$")
ax[0].axvline(r_MOND/kpc, color='red', ls=':', alpha=0.6, label=r"$r_{\rm MOND}$")
ax[0].set_xlabel("r [kpc]"); ax[0].set_ylabel(r"$|\theta'|$")
ax[0].set_title("DGSAT-I Plummer: scalar gradient")
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

ax[1].loglog(r_kpc, ratio, label=r"$(\theta'_A/\theta'_B)^2 = \mu^2$")
ax[1].axhline(1.0, color='k', lw=0.5)
ax[1].axhline(0.9, color='orange', ls='--', lw=0.7, label='10% band')
ax[1].axhline(1.1, color='orange', ls='--', lw=0.7)
ax[1].axvline(r_eff/kpc, color='gray', ls=':')
ax[1].axvline(r_MOND/kpc, color='red', ls=':', alpha=0.6)
ax[1].set_xlabel("r [kpc]"); ax[1].set_ylabel(r"$(\theta'_A/\theta'_B)^2$")
ax[1].set_title("Ratio: naive DM form vs exact spherical")
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig("paper_III/theta_prime_dgsat.png", dpi=130)

with open("paper_III/theta_prime_dgsat_report.txt", "w", encoding="utf-8") as f:
    f.write("Paper III - Direct scalar-field integration: DGSAT-I\n")
    f.write("="*64 + "\n\n")
    f.write("CONVENTIONS\n")
    f.write("  Spherical AQUAL: r^2 mu(g/a0) theta' = G M(r)\n")
    f.write("  with mu(x) = x/sqrt(1+x^2), nu(y) = sqrt(0.5+0.5*sqrt(1+4/y^2))\n")
    f.write("  =>  theta'(r) = g_N(r) / [mu(g_obs/a0) * a0]   (dimensionless form)\n\n")
    f.write("PROFILE\n")
    f.write(f"  Plummer M = 3e8 Msun, r_eff = 2.1 kpc (DGSAT-I; Paper II Table I)\n")
    f.write(f"  r_MOND = sqrt(GM/a0) = {r_MOND/kpc:.3f} kpc\n")
    f.write(f"  epsilon = r_eff/r_MOND = {r_eff/r_MOND:.3f}\n\n")
    f.write("TWO FORMS COMPARED\n")
    f.write("  A (naive deep-MOND):   theta'_A = g_N / a0\n")
    f.write("  B (exact spherical):   theta'_B = g_N / (mu * a0)\n")
    f.write("  Ratio (A/B)^2 = mu^2(g_obs/a0).\n\n")
    f.write("AGREEMENT REGIMES (DGSAT-I)\n")
    f.write(f"  Worst |1 - mu^2| = {worst_diff_pct:.1f} %  at r = {r_kpc[diff.argmax()]:.2f} kpc\n")
    f.write(f"  Median|1 - mu^2| = {median_diff_pct:.1f} %\n")
    if agree_mask.any():
        first = r_kpc[np.argmax(agree_mask)]
        last  = r_kpc[len(agree_mask)-1 - np.argmax(agree_mask[::-1])]
        f.write(f"  10%-agreement band: NONE for r in [{r_kpc[0]:.2f}, {r_kpc[-1]:.2f}] kpc\n"
                if not agree_mask.any() else
                f"  10%-agreement band: r in [{first:.2f}, {last:.2f}] kpc\n")
    else:
        f.write(f"  10%-agreement band: NONE in [{r_kpc[0]:.2f}, {r_kpc[-1]:.2f}] kpc\n")
        f.write("  (Note: for DGSAT-I, g_N << a0 everywhere, so mu << 1 throughout.\n"
                "   The naive form A SEVERELY UNDERESTIMATES theta' for extended deep-MOND profiles.)\n")
    f.write("\nSAMPLE TABLE  (r in kpc)\n")
    f.write("  r       y=g_N/a0    mu(g/a0)   theta'_A     theta'_B     A/B\n")
    for i in [i_inner, i_eff, i_3eff, len(r)-1]:
        tA = np.sqrt(thetaA2[i]); tB = np.sqrt(thetaB2[i])
        f.write(f"  {r_kpc[i]:6.2f}  {y[i]:.3e}  {mu_arr[i]:.3e}  "
                f"{tA:.3e}  {tB:.3e}  {tA/tB:.3e}\n")
    f.write("\nINTERPRETATION\n")
    f.write("  In the deep-MOND regime g_N << a0, mu ~ g/a0 ~ sqrt(g_N/a0) << 1, so\n")
    f.write("  theta'_B = g_N/(mu*a0) ~ sqrt(g_N/a0)  (matches the AQUAL solution).\n")
    f.write("  But the naive form theta'_A = g_N/a0 = y << 1 is QUADRATICALLY smaller.\n\n")
    f.write("  Therefore (theta'_A)^2 = y^2 vs (theta'_B)^2 = y\n")
    f.write("  -> They differ by a factor y everywhere in deep-MOND.\n")
    f.write("  -> They agree only when y >> 1 (Newtonian regime), which DGSAT-I never\n")
    f.write("     reaches: y(r_eff) = {:.2e}.\n".format(y[i_eff]))
    f.write("\nCONCLUSION FOR PAPER III\n")
    f.write("  The identification (theta')^2 = g_N^2/a0^2 used in the eta_diagnosis\n")
    f.write("  hypothesis (and implicitly in the source S of the literal Paper I\n")
    f.write("  derivation) is the WRONG closure for extended deep-MOND systems.\n")
    f.write("  The exact AQUAL gradient gives (theta')^2 = g_N/a0 in deep-MOND, i.e.\n")
    f.write("  ONE power of (g_N/a0), not two.  This is the structural fix that any\n")
    f.write("  extended-source derivation of the slip must respect.\n")

print(f"\nWorst |1-mu^2| = {worst_diff_pct:.1f} %, median {median_diff_pct:.1f} %")
print(f"y(r_eff) = {y[i_eff]:.2e}, mu(r_eff) = {mu_arr[i_eff]:.2e}")
