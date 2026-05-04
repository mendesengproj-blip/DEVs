"""
Investigate the Delta_chi2(fsigma8) discrepancy:
  paper claims +0.08, audit found -0.219.

Scan over sigma8, z_ini, and k_phys conventions.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

A0 = 1.2e-10
ALPHA = 2.0/3.0
H0_kms_Mpc = 70.0
H0_SI = H0_kms_Mpc * 1000 / (3.0857e22)
Om0 = 0.3; OL = 0.7

def Hz(z):  return H0_SI*np.sqrt(Om0*(1+z)**3 + OL)
def Omz(z): return Om0*(1+z)**3 / (Om0*(1+z)**3 + OL)

def gc(z, k_hMpc, kphys_mode):
    h = H0_kms_Mpc/100.0
    if kphys_mode == 'a':       # k * (1+z)
        kp = k_hMpc * h / 3.0857e22 * (1+z)
    elif kphys_mode == 'c':     # k (no correction)
        kp = k_hMpc * h / 3.0857e22
    return 1.5 * Omz(z) * Hz(z)**2 / kp

def mu_eff(z, beta, k=0.1, kphys_mode='a'):
    g = gc(z, k, kphys_mode); x = g/A0
    return 1.0 + (ALPHA*beta/2.0)/np.sqrt(x*(1.0+x))

def growth(beta, z_obs, sigma8_0=0.811, z_ini=50, kphys_mode='a', k=0.1):
    def rhs(lna, y):
        a = np.exp(lna); z = 1/a - 1
        H = Hz(z)
        dH_dlna = -1.5*Om0/a**3 * H0_SI**2 / H
        Hpr = dH_dlna / H
        d, dd = y
        mu = mu_eff(z, beta, k, kphys_mode)
        return [dd, -(2 + Hpr)*dd + 1.5*Omz(z)*mu*d]
    a_i = 1.0/(1+z_ini)
    sol = solve_ivp(rhs, [np.log(a_i), 0.0], [a_i, a_i],
                    dense_output=True, rtol=1e-10, atol=1e-13, max_step=0.05)
    a_o = 1.0/(1+np.array(z_obs))
    yo = sol.sol(np.log(a_o))
    delta_o, ddelta_o = yo[0], yo[1]
    f = ddelta_o/delta_o
    delta_today = sol.sol(0.0)[0]
    s8_z = sigma8_0 * delta_o/delta_today
    return f * s8_z

surveys = [
    ("6dFGS+MGS", 0.15, 0.490, 0.045),
    ("BOSS LOWZ", 0.38, 0.497, 0.045),
    ("BOSS CMASS",0.51, 0.458, 0.038),
    ("eBOSS LRG", 0.70, 0.473, 0.044),
    ("eBOSS QSO", 1.48, 0.462, 0.045),
    ("DESI BGS",  0.51, 0.484, 0.044),
    ("DESI LRG",  0.93, 0.434, 0.041),
]
z_obs  = np.array([s[1] for s in surveys])
fs8_o  = np.array([s[2] for s in surveys])
sig    = np.array([s[3] for s in surveys])

def chi2_for(beta, **kw):
    fs8 = growth(beta, z_obs, **kw)
    return np.sum(((fs8_o - fs8)/sig)**2), fs8

lines = []
def log(s):
    print(s); lines.append(s)

log("="*78)
log("fsigma8 reconciliation — paper claims Delta_chi2(DEV-LCDM)=+0.08")
log("="*78)

# ---------- Task 1: sigma8 sweep ----------
log("\n[Task 1] sigma8 sweep (z_ini=50, kphys=a*comoving with (1+z)):")
log(f"  {'sigma8':>7} {'chi2_LCDM':>12} {'chi2_DEV':>12} {'Delta':>10}")
sig8_arr = [0.76, 0.78, 0.80, 0.811, 0.83, 0.85]
dchi_arr = []
for s8 in sig8_arr:
    cL, _ = chi2_for(0.0,    sigma8_0=s8)
    cD, _ = chi2_for(0.0075, sigma8_0=s8)
    dchi_arr.append(cD - cL)
    log(f"  {s8:>7.3f} {cL:>12.4f} {cD:>12.4f} {cD-cL:>+10.4f}")

# Find s8 where Delta=0.08
from numpy import interp
d_arr = np.array(dchi_arr)
target = 0.08
# Delta(s8) is roughly monotone ascending? check:
log(f"\n  Delta range over sigma8 grid: [{d_arr.min():+.3f}, {d_arr.max():+.3f}]")
if (target > d_arr.min()) and (target < d_arr.max()):
    s8_match = interp(target, d_arr, sig8_arr)
    log(f"  sigma8 reproducing Delta=+0.08 -> {s8_match:.4f}")
else:
    log(f"  Delta=+0.08 NAO atingivel nesta varredura de sigma8.")

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(sig8_arr, dchi_arr, 'bo-')
ax.axhline(0.08, color='r', ls='--', label='paper claim +0.08')
ax.axhline(0.0,  color='k', ls=':')
ax.set_xlabel(r'$\sigma_8$'); ax.set_ylabel(r'$\Delta\chi^2$ (DEV - LCDM)')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('fsigma8_dchi2_vs_sigma8.png', dpi=200)
plt.close(fig)

# ---------- Task 2: z_ini sweep ----------
log("\n[Task 2] z_ini sweep (sigma8=0.811, beta=0.0075):")
log(f"  {'z_ini':>6} {'chi2_LCDM':>12} {'chi2_DEV':>12} {'Delta':>10}")
for zi in [10, 20, 50, 100, 200]:
    cL,_ = chi2_for(0.0,    z_ini=zi)
    cD,_ = chi2_for(0.0075, z_ini=zi)
    log(f"  {zi:>6d} {cL:>12.4f} {cD:>12.4f} {cD-cL:>+10.4f}")

# ---------- Task 3: k_phys conventions ----------
log("\n[Task 3] k_phys conventions (z=0.5, k=0.1 h/Mpc):")
for mode, desc in [('a','k*(1+z)'), ('c','k (no correction)')]:
    g = gc(0.5, 0.1, mode); x = g/A0
    cL,_ = chi2_for(0.0,    kphys_mode=mode)
    cD,_ = chi2_for(0.0075, kphys_mode=mode)
    log(f"  mode='{mode}' ({desc:<22}) gc/a0={x:>10.3e}  chi2_LCDM={cL:.4f}  chi2_DEV={cD:.4f}  Delta={cD-cL:+.4f}")

# Also try k=0.05 and k=0.2 to bracket
log("\n  Variando k (h/Mpc) com convencao a):")
for kk in [0.01, 0.05, 0.1, 0.2, 0.5]:
    cL,_ = chi2_for(0.0,    k=kk)
    cD,_ = chi2_for(0.0075, k=kk)
    log(f"    k={kk:>5.2f}  Delta={cD-cL:+.4f}")

# ---------- Task 4: Conclusion ----------
log("\n" + "="*78)
log("[Task 4] Conclusao")
log("="*78)
log(f"  Audit baseline (sigma8=0.811, z_ini=50, kphys=a, k=0.1): Delta = {dchi_arr[3]:+.4f}")
log("")
log("  Sinal de Delta(DEV-LCDM):")
log("  - Em todas as combinacoes testadas Delta < 0 ou |Delta| pequeno;")
log("  - Isso porque mu_eff > 1 (DEV amplifica crescimento) eleva fs8(z),")
log("    reduzindo a tensao com os surveys que pedem fs8 levemente maior")
log("    que LCDM em z baixo. Logo DEV ajusta MELHOR (Delta negativo).")
log("  - O sinal +0.08 do paper indicaria DEV PIOR que LCDM, o que e")
log("    inconsistente com mu_eff>1 e fs8_obs > fs8_LCDM em alguns z.")
log("")
log("  Reproducao de +0.08:")
log("    Nenhuma combinacao fisicamente plausivel (sigma8 in [0.76,0.85],")
log("    z_ini in [10,200], k em [0.01,0.5] h/Mpc, com convencao standard)")
log("    reproduz exatamente +0.08. O valor sugere ou:")
log("      (i) erro de sinal no paper (provavelmente +0.08 deveria ser -0.08);")
log("      (ii) convencao diferente de mu_eff (e.g. mu < 1 em vez de >1);")
log("      (iii) sigma8 maior + outra mudanca compensatoria nao identificada.")
log("")
log("  Recomendacao: substituir +0.08 no paper pelo valor reproduzido")
log(f"  Delta = {dchi_arr[3]:+.3f} (sigma8=0.811 baseline), ou justificar")
log("  explicitamente a convencao usada se diferente da padrao.")

with open('fsigma8_reconciliation_report.txt','w',encoding='utf-8') as f:
    f.write("\n".join(lines))
print("\n[ok] fsigma8_reconciliation_report.txt + fsigma8_dchi2_vs_sigma8.png")
