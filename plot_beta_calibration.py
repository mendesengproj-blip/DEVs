import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from theory import eta_dev, A0
from calibrate_beta import ETA_CONSTRAINTS, fit_beta

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})

LABEL_MAP = {
    'SDSS_galaxies':  'SDSS galaxies',
    'CFHTLenS':       'CFHTLenS',
    'MACS_J1206':     'MACS J1206',
    'A1689_lens_dyn': 'A1689 (lens+dyn)',
    'Coma_outer':     'Coma outer (est.)',
}

os.makedirs("figures", exist_ok=True)
res = fit_beta()
beta_best, beta_lo, beta_hi = res['beta_best'], res['beta_lo'], res['beta_hi']

x_grid = np.logspace(-3.5, 1.5, 400)
g_grid = x_grid * A0
eta_best = eta_dev(g_grid, beta=beta_best) - 1
eta_lo = eta_dev(g_grid, beta=beta_lo) - 1
eta_hi = eta_dev(g_grid, beta=beta_hi) - 1

fig, ax = plt.subplots(figsize=(8.5, 6))
ax.fill_between(x_grid, eta_lo, eta_hi, color='steelblue', alpha=0.25,
                label=fr'1$\sigma$: $\beta\in[{beta_lo:.4f},{beta_hi:.4f}]$')
ax.plot(x_grid, eta_best, 'k-', lw=2.2,
        label=fr'DEV: $\beta_{{\rm best}}={beta_best:.4f}$, $\alpha=2/3$')

for name, x, eta_obs, eta_err, ref in ETA_CONSTRAINTS:
    ax.errorbar(x, eta_obs - 1, yerr=eta_err, fmt='o', ms=8,
                color='darkred', ecolor='darkred', capsize=4, zorder=5)
    ax.annotate(LABEL_MAP.get(name, name), (x, eta_obs - 1),
                xytext=(8, 6), textcoords='offset points', fontsize=8.5)

udgs = [('NGC1052-DF2',0.048),('NGC1052-DF4',0.068),('DF44',0.016),
        ('DGSAT-I',0.0025),('VCC1287',0.017),('DF17',0.006)]
for name, x in udgs:
    eta_m1 = float(eta_dev(x*A0, beta=beta_best)) - 1
    ax.scatter(x, eta_m1, marker='*', s=140, color='gold',
               edgecolors='black', lw=0.8, zorder=6)
    ax.annotate(name, (x, eta_m1), xytext=(6, -12),
                textcoords='offset points', fontsize=8, color='darkgoldenrod')

ax.axhspan(0.01, 0.05, alpha=0.10, color='green',
           label=r'Euclid sensitivity ($\sim$1--5\%)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$x = g/a_0$')
ax.set_ylabel(r'$\eta - 1 \equiv \Phi/\Psi - 1$')
ax.set_title(r'DEV gravitational slip: calibration of $\beta$')
ax.text(0.02, 0.02, 'Coma outer: estimated constraint',
        transform=ax.transAxes, fontsize=8, alpha=0.7, style='italic')
ax.legend(loc='lower left', framealpha=0.95, fontsize=9)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(3e-4, 30); ax.set_ylim(1e-4, 3)

out = "figures/fig_beta_calibration.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] {out}")
print(f"beta_best = {beta_best:.4f}  chi2_red = {res['chi2_red']:.3f}")
