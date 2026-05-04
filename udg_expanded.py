"""
Expansão da Tabela III: perfil radial de eta para DGSAT-I
e tabela expandida de UDGs com forecast Euclid.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from theory import A0, G_NEWTON, KPC_TO_M, MSUN, nu_dev, eta_dev

BETA = 0.0075
ALPHA = 2.0/3.0


def g_newton_pointmass(M_msun, r_kpc):
    return G_NEWTON * (M_msun*MSUN) / (r_kpc*KPC_TO_M)**2


def g_newton_plummer(M_msun, r_kpc, r_eff_kpc):
    r_m = r_kpc * KPC_TO_M
    a_m = r_eff_kpc * KPC_TO_M
    M_enc = M_msun*MSUN * r_m**3 / (r_m**2 + a_m**2)**1.5
    return G_NEWTON * M_enc / r_m**2


def eta_minus_1(g_obs, beta=BETA, alpha=ALPHA):
    x = g_obs / A0
    return alpha * beta / np.sqrt(x*(1.0+x))


# ============================================================
# Tarefa 1 — Perfil radial DGSAT-I
# ============================================================
M_DGSAT = 3.0e8
REFF_DGSAT = 4.7

r_arr = np.linspace(0.1*REFF_DGSAT, 5.0*REFF_DGSAT, 400)
gN = g_newton_plummer(M_DGSAT, r_arr, REFF_DGSAT)
gobs = nu_dev(gN/A0) * gN
eta_m1 = eta_minus_1(gobs)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(r_arr, eta_m1*100, 'b-', lw=2, label=r'DEV $\eta-1$ (DGSAT-I, Plummer)')
ax.axvline(REFF_DGSAT, color='k', ls=':', label=r'$r_{\rm eff}=4.7$ kpc')
ax.axhspan(1.0, 5.0, color='green', alpha=0.2, label='Euclid sensitivity (1-5%)')
ax.set_xlabel('r [kpc]')
ax.set_ylabel(r'$\eta(r)-1$ [%]')
ax.set_title(r'Perfil radial de slip gravitacional — DGSAT-I ($\beta=0.0075$)')
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('DGSAT_eta_profile.png', dpi=300)
plt.close(fig)
print("[ok] DGSAT_eta_profile.png")

# ============================================================
# Tarefa 2 — Tabela expandida
# ============================================================
udgs = [
    # (name, M_star, r_eff, ref, is_original)
    ("NGC1052-DF2",  2.0e8, 2.2, "vanDokkum+2018", True),
    ("NGC1052-DF4",  1.5e8, 2.0, "vanDokkum+2019", True),
    ("DF44",         3.0e8, 4.3, "vanDokkum+2016", True),
    ("DGSAT-I",      3.0e8, 4.7, "Martinez-Delgado+2016", True),
    ("VCC1287",      1.0e8, 2.8, "Beasley+2016", True),
    ("DF17",         2.0e8, 3.5, "vanDokkum+2017", True),
    ("Dragonfly17",  5.0e7, 3.1, "Merritt+2016", False),
    ("Dragonfly44",  3.0e8, 4.3, "vanDokkum+2016", False),
    ("UDG1-Coma",    8.0e7, 2.5, "Koda+2015", False),
    ("LSBG-285",     2.0e8, 5.2, "Tanoglidis+2021", False),
    ("AntliaII",     1.5e9, 2.9, "Torrealba+2019", False),
    ("CraterII",     3.7e8, 1.1, "Caldwell+2017", False),
]

SIGMA_ETA = 0.05
rows = []
for name, Ms, reff, ref, is_orig in udgs:
    gN_pt = g_newton_pointmass(Ms, reff)
    x_N = gN_pt/A0
    gobs_pt = nu_dev(x_N) * gN_pt
    x = gobs_pt/A0
    em1 = ALPHA*BETA/np.sqrt(x*(1.0+x))
    snr = em1 / SIGMA_ETA
    detect = "Sim" if em1*100 > 1.0 else "Nao"
    if x > 1.0:
        regime = "Newtoniano (slip suprimido)"
    elif x > 0.1:
        regime = "Transicao"
    else:
        regime = "MOND profundo"
    rows.append({
        'name': name, 'M_star': Ms, 'r_eff': reff, 'ref': ref,
        'is_orig': is_orig,
        'g_over_a0': x, 'g_over_a0_N': x_N,
        'eta_minus_1': em1, 'eta_minus_1_pct': em1*100,
        'snr': snr, 'detect': detect, 'regime': regime,
    })

# CSV
with open('udg_results.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['name','M_star_Msun','r_eff_kpc','ref','g/a0','eta-1','eta-1_pct','SNR','Euclid_detect','regime'])
    for r in rows:
        w.writerow([r['name'], r['M_star'], r['r_eff'], r['ref'],
                    f"{r['g_over_a0']:.4f}", f"{r['eta_minus_1']:.5f}",
                    f"{r['eta_minus_1_pct']:.3f}", f"{r['snr']:.3f}",
                    r['detect'], r['regime']])
print("[ok] udg_results.csv")

# ============================================================
# Tarefa 3 — Plot expandido
# ============================================================
x_curve = np.logspace(-4, 1.5, 500)
def curve(x, beta): return ALPHA*beta/np.sqrt(x*(1.0+x))*100

fig, ax = plt.subplots(figsize=(9,6))
ax.plot(x_curve, curve(x_curve, 0.0075), 'k-',  lw=2, label=r'DEV $\beta=0.0075$')
ax.plot(x_curve, curve(x_curve, 0.005),  'k--', lw=1, label=r'$\beta=0.005$')
ax.plot(x_curve, curve(x_curve, 0.010),  'k:',  lw=1, label=r'$\beta=0.010$')
ax.axhspan(1.0, 5.0, color='green', alpha=0.2, label='Euclid (1-5%)')

for r in rows:
    color = 'blue' if r['is_orig'] else 'red'
    ax.scatter(r['g_over_a0'], r['eta_minus_1_pct'], c=color, s=60,
               edgecolors='k', zorder=5)
    ax.annotate(r['name'], (r['g_over_a0'], r['eta_minus_1_pct']),
                fontsize=7, xytext=(4,4), textcoords='offset points')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$g/a_0$')
ax.set_ylabel(r'$\eta-1$ [%]')
ax.set_title('UDG expanded sample — DEV gravitational slip prediction')
ax.legend(loc='lower left', fontsize=8)
ax.grid(alpha=0.3, which='both')
ax.scatter([],[],c='blue',edgecolors='k',label='original Tabela III')
ax.scatter([],[],c='red', edgecolors='k',label='added')
fig.tight_layout()
fig.savefig('UDG_eta_expanded.png', dpi=300)
plt.close(fig)
print("[ok] UDG_eta_expanded.png")

# ============================================================
# Tarefa 4 — Detectabilidade
# ============================================================
n_indiv = sum(1 for r in rows if r['snr'] > 3)
n_stack = sum(1 for r in rows if r['snr'] > 1)
snr_stack = np.sqrt(sum(r['snr']**2 for r in rows))

# ============================================================
# Tarefa 5 — LaTeX table
# ============================================================
lines = [
    r"\begin{table*}[ht]",
    r"\centering",
    r"\caption{Predições DEV expandidas para UDGs ($\beta=0.0075$, $\alpha=2/3$).}",
    r"\label{tab:udg_expanded}",
    r"\begin{tabular}{lcccccccl}",
    r"\hline",
    r"System & $M_\star$ [$M_\odot$] & $r_{\rm eff}$ [kpc] & $g/a_0$ & $\eta-1$ [\%] & SNR & Detect.? & Ref. \\",
    r"\hline",
]
for r in rows:
    lines.append(
        f"{r['name']} & {r['M_star']:.1e} & {r['r_eff']:.1f} & "
        f"{r['g_over_a0']:.3f} & {r['eta_minus_1_pct']:.2f} & "
        f"{r['snr']:.2f} & {r['detect']} & {r['ref']} \\\\"
    )
lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
with open('table_udg_expanded.tex','w') as f:
    f.write('\n'.join(lines))
print("[ok] table_udg_expanded.tex")

# ============================================================
# Report
# ============================================================
report = []
report.append("="*70)
report.append("RELATORIO — Tabela III Expandida (DEV slip em UDGs)")
report.append("="*70)
report.append(f"Parametros: beta={BETA}, alpha={ALPHA:.4f}, a0={A0:.2e} m/s^2")
report.append(f"sigma_eta assumido por sistema: {SIGMA_ETA}")
report.append("")
report.append(f"{'Sistema':<15}{'g/a0':>10}{'eta-1[%]':>12}{'SNR':>8}{'Detect':>10}  Regime")
report.append("-"*80)
for r in rows:
    flag = "*" if r['is_orig'] else " "
    report.append(f"{flag}{r['name']:<14}{r['g_over_a0']:>10.4f}{r['eta_minus_1_pct']:>12.3f}"
                  f"{r['snr']:>8.2f}{r['detect']:>10}  {r['regime']}")
report.append("")
report.append(f"Sistemas detectaveis individualmente (SNR>3): {n_indiv}/{len(rows)}")
report.append(f"Sistemas que contribuem ao stack (SNR>1):    {n_stack}/{len(rows)}")
report.append(f"SNR combinado (stacking) = {snr_stack:.2f}")
report.append("")
report.append("Notas criticas:")
for r in rows:
    if r['g_over_a0'] > 1.0:
        report.append(f"  - {r['name']}: g/a0={r['g_over_a0']:.2f} > 1 -> regime Newtoniano, "
                      f"slip fortemente suprimido (eta-1={r['eta_minus_1_pct']:.3f}%). "
                      "NAO recomendado como alvo Euclid.")
    elif r['g_over_a0'] > 0.1:
        report.append(f"  - {r['name']}: g/a0={r['g_over_a0']:.2f} em transicao; "
                      "formula completa usada (nao deep-MOND).")
report.append("")
report.append("Caveat: aproximacao de massa pontual no r_eff superestima g_N "
              "comparado a perfil estendido (Plummer/Sersic). Para sistemas com "
              "perfil bem resolvido (DGSAT-I, DF44) o slip real no r_eff e "
              "ligeiramente maior que o tabulado. Ver DGSAT_eta_profile.png.")

with open('udg_report.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(report))
print("[ok] udg_report.txt")
print()
print('\n'.join(report))
