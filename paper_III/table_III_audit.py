"""Audit + corrected Table III + Fisher SNR."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'paper_I'))
from theory import A0, G_NEWTON, KPC_TO_M, MSUN, nu_dev

BETA = 0.0075
ALPHA = 2.0/3.0

# ---------- TASK 1: what r_eff would give g/a0 = 0.0025 for DGSAT-I? ----------
def r_for_gn_over_a0(M_msun, x_target):
    gN = x_target * A0
    M_kg = M_msun * MSUN
    r_m = np.sqrt(G_NEWTON * M_kg / gN)
    return r_m / KPC_TO_M

r_implied = r_for_gn_over_a0(3.0e8, 0.0025)
print("="*72)
print("TASK 1 — Reverse-engineer r_eff from Paper I Table III value g/a0=0.0025")
print("="*72)
print(f"  M_star          = 3e8 Msun")
print(f"  g_N/a0 (table)  = 0.0025")
print(f"  Implied r_eff   = {r_implied:.2f} kpc")
print(f"  Real DGSAT-I    = 4.7 kpc (Janowiecki+15)  OR  2.1 kpc (Mihos+15)")
print(f"  ==> Table III used r_eff ~ {r_implied:.1f} kpc, which is {r_implied/4.7:.1f}x")
print(f"      the photometric r_eff from Janowiecki+15 (4.7 kpc).\n")

# ---------- TASK 2: corrected table ----------
UDGs = [
    ("NGC1052-DF2", 2.0e8, 2.2),
    ("NGC1052-DF4", 1.5e8, 2.0),
    ("DF44",        3.0e8, 4.3),
    ("DGSAT-I",     3.0e8, 4.7),
    ("VCC1287",     1.0e8, 2.8),
    ("DF17",        2.0e8, 3.5),
]

def eta_m1(x):  # x = whatever the chosen argument is
    return ALPHA * BETA / np.sqrt(x*(1.0+x))

print("="*72)
print("TASK 2 — Corrected Table III (two conventions side-by-side)")
print("="*72)
print(f"{'system':<13} {'M*':>9} {'r_eff':>6} {'g_N/a0':>9} {'g_obs/a0':>10}"
      f" {'eta-1 [g_N]':>12} {'eta-1 [g_obs]':>14} {'r_eff/r_M':>10}")
rows = []
for name, M, reff in UDGs:
    gN = G_NEWTON * M*MSUN / (reff*KPC_TO_M)**2
    xN = gN/A0
    xO = nu_dev(xN)*xN          # g_obs/a0
    e_N = eta_m1(xN)*100
    e_O = eta_m1(xO)*100
    rMOND = np.sqrt(G_NEWTON*M*MSUN/A0)/KPC_TO_M
    eps = reff/rMOND
    rows.append((name, M, reff, xN, xO, e_N, e_O, eps))
    print(f"{name:<13} {M:9.2e} {reff:6.2f} {xN:9.4f} {xO:10.4f}"
          f" {e_N:12.3f} {e_O:14.3f} {eps:10.2f}")

# Identify the canonical convention. Reproduce existing Table III to confirm.
print("\nReproduce existing Table III using x = g_N/a0:")
print(f"  DF2:    table=2.2,  computed_gN={rows[0][5]:.2f}")
print(f"  DGSAT-I (with r_eff=11.8): "
      f"x_N=0.0025 -> eta-1={eta_m1(0.0025)*100:.2f}  (table says 9.9 -- match)")
print("==> existing Table III convention: x = g_N/a0\n")

# ---------- TASK 3: corrected LaTeX table ----------
tex = []
tex.append(r"% Paper III -- Corrected Table III (replaces tab:udg in Paper I)")
tex.append(r"\begin{table}[h]")
tex.append(r"\caption{DEV gravitational slip predictions for known UDGs,")
tex.append(r"recomputed with published photometric parameters and using the")
tex.append(r"argument $x \equiv g_N/\aO$ as in Paper~I, Sec.~III.")
tex.append(r"Two conventions are tabulated: $\eta-1$ evaluated at the")
tex.append(r"Newtonian $g_N$ (column~5, the original Paper~I convention) and")
tex.append(r"at the observed $g_{\rm obs}=\nu(g_N/\aO)\,g_N$ (column~6,")
tex.append(r"the convention adopted in Paper~III).}")
tex.append(r"\label{tab:udg}")
tex.append(r"\begin{ruledtabular}")
tex.append(r"\begin{tabular}{lcccccc}")
tex.append(r"System & $M_\star$ & $r_{\rm eff}$ & $g_N/\aO$ & $g_{\rm obs}/\aO$"
           r" & $(\eta-1)_{g_N}$\,[\%] & $(\eta-1)_{g_{\rm obs}}$\,[\%] \\")
tex.append(r"       & ($10^8 M_\odot$) & (kpc) & & & & \\")
tex.append(r"\hline")
for name, M, reff, xN, xO, eN, eO, eps in rows:
    tex.append(f"{name} & {M/1e8:.1f} & {reff:.1f} & {xN:.4f} & {xO:.4f}"
               f" & {eN:.2f} & {eO:.2f} \\\\")
tex.append(r"\end{tabular}")
tex.append(r"\end{ruledtabular}")
tex.append(r"{\small Note: the previous version of this table (dev\_paper\_I\_corrected.tex)")
tex.append(r"used $r_{\rm eff}=11.8$\,kpc for DGSAT-I, which yielded")
tex.append(r"$g_N/\aO=0.0025$ and $\eta-1=9.9\%$. With the photometric value")
tex.append(r"$r_{\rm eff}=4.7$\,kpc (Janowiecki+15), $g_N/\aO=0.016$ and the")
tex.append(r"prediction collapses to $\eta-1\simeq 4.0\%$ (or $1.3\%$ in the")
tex.append(r"$g_{\rm obs}$ convention). All six UDGs lie in the deep-MOND regime")
tex.append(r"and predict $\eta-1\sim 1$--$5\%$, near the Euclid threshold.}")
tex.append(r"\end{table}")

with open(os.path.join(os.path.dirname(__file__), "table_III_corrected.tex"), "w") as f:
    f.write("\n".join(tex))
print("Wrote table_III_corrected.tex\n")

# ---------- TASK 5: corrected abstract ----------
abstract = r"""% Paper III -- corrected abstract fragment for Paper I
%
% REPLACE in dev_paper_I_corrected.tex around line 60:
%
% OLD:
%   The theory predicts gravitational slip of $2$--$10\%$ in
%   ultra-diffuse galaxies (UDGs) with $g \ll \aO$, a unique
%   ...
%
% NEW:
The theory predicts a gravitational slip
$\eta-1 \simeq 1$--$5\%$ in ultra-diffuse galaxies (UDGs)
with $g_N \ll \aO$, near the sensitivity threshold of the
Euclid satellite ($\delta\eta \sim 1$--$5\%$). Detection
will require either dedicated deep weak-lensing follow-up
of individual high-priority UDGs (DF44, DGSAT-I, VCC1287,
DF17), or stacking of $\mathcal{O}(10^2)$ UDGs, both of
which are within the reach of Euclid and the Roman Space
Telescope. The previously reported figure of
$\eta-1 \simeq 9.9\%$ for DGSAT-I traces back to a
non-photometric value of $r_{\rm eff}$ used in an earlier
draft of Table~\ref{tab:udg} and is corrected in Paper~III.
"""
with open(os.path.join(os.path.dirname(__file__), "abstract_corrected.tex"), "w") as f:
    f.write(abstract)
print("Wrote abstract_corrected.tex\n")

# ---------- TASK 6: Fisher SNR ----------
print("="*72)
print("TASK 6 — Fisher SNR re-evaluation (g_N convention, matches Paper I)")
print("="*72)
eN_arr = np.array([r[5] for r in rows])/100.0
eO_arr = np.array([r[6] for r in rows])/100.0
med_N = np.median(eN_arr)
med_O = np.median(eO_arr)

def snr(N, eta_m1, sigma_eta):
    return np.sqrt(N) * eta_m1 / sigma_eta

print(f"  median(eta-1) [g_N convention]    = {med_N*100:.3f} %")
print(f"  median(eta-1) [g_obs convention]  = {med_O*100:.3f} %\n")
for label, eta in [("g_N", med_N), ("g_obs", med_O)]:
    s6   = snr(6,   eta, 0.05)
    s300 = snr(300, eta, 0.02)
    print(f"  [{label}]  N=6,   sigma=0.05 -> SNR = {s6:5.2f}")
    print(f"  [{label}]  N=300, sigma=0.02 -> SNR = {s300:5.2f}")

print("\nPrevious paper claim: SNR_Euclid ~ 8 with N=300 and median eta-1 ~ 5%.")
print("With corrected medians SNR(N=300, sigma=0.02) drops accordingly.\n")
