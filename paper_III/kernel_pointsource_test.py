"""
Paper III - Poisson kernel integral test
Tests whether the Green's function convolution of S_correto = (2 beta/3) g_N^2 / (mu^2 c^2)
recovers the Paper I analytical eta-1 formula in the point-source limit.

Units: galactic
  [length]       = kpc
  [velocity]     = km/s
  [acceleration] = (km/s)^2 / kpc
  [mass]         = Msun
  G  = 4.302e-3   kpc (km/s)^2 / Msun
  c  = 2.998e5    km/s
  a0 = 3.857e-11  (km/s)^2 / kpc
"""

import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
import matplotlib.pyplot as plt
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# -------- constants (galactic units) --------
G    = 4.302e-3
C    = 2.998e5
C2   = C * C
A0   = 3702.0   # (km/s)^2/kpc, correct conversion of 1.2e-10 m/s^2 (see commit e0f5f39)
BETA = 0.0075

# -------- Plummer profile --------
def M_plummer(r, M, a):
    return M * r**3 / (r**2 + a**2)**1.5

def gN_plummer(r, M, a):
    return G * M_plummer(r, M, a) / r**2

# -------- DEV interpolating functions --------
def nu(y):
    return np.sqrt(0.5 + 0.5 * np.sqrt(1.0 + 4.0 / y**2))

def mu_from_x(x):
    return x / np.sqrt(1.0 + x*x)

# -------- Source S_correto --------
def S_correto(r, M, a):
    gN = gN_plummer(r, M, a)
    x  = gN / A0
    mu = mu_from_x(x)
    return (2.0 * BETA / 3.0) * gN**2 / (mu**2 * C2)

# -------- Poisson Green's function (3D spherical) --------
# Solution of  (1/r^2) d/dr( r^2 d f / dr ) = S(r),  f -> 0 at infinity:
#   f(r) = - (1/r) * int_0^r r'^2 S(r') dr'  -  int_r^inf r' S(r') dr'
def poisson_solve(r_grid, S_grid):
    # cumulative integrals on the grid
    inner = cumulative_trapezoid(r_grid**2 * S_grid, r_grid, initial=0.0)
    # outer = int_r^inf r' S(r') dr'  =  total - cumulative(r' S)
    cum_outer = cumulative_trapezoid(r_grid * S_grid, r_grid, initial=0.0)
    total_outer = cum_outer[-1]
    outer = total_outer - cum_outer
    f = -inner / r_grid - outer
    return f

# -------- Newtonian-MOND potential Psi(r) = -int_r^inf g_obs dr' --------
def Psi_of_r(r_grid, gobs_grid):
    cum = cumulative_trapezoid(gobs_grid, r_grid, initial=0.0)
    return -(cum[-1] - cum)  # Psi(r) = -(int_r^inf g_obs dr')

# -------- analytical eta from Paper I --------
def eta_analytical(gobs):
    x = gobs / A0
    return 1.0 + (2.0/3.0) * BETA / np.sqrt(x * (1.0 + x))

# ------------------------------------------------------------------
# TASK 2: quasi-point source validation
# ------------------------------------------------------------------
def run_pointsource():
    M    = 1.0e10
    a    = 0.01
    rMOND = np.sqrt(G * M / A0)
    print(f"[POINT SOURCE] M={M:.2e} Msun, r_eff={a} kpc, r_MOND={rMOND:.3f} kpc")

    r_max_grid = max(500.0, 200.0 * rMOND)
    r = np.logspace(np.log10(0.001), np.log10(r_max_grid), 4000)
    gN   = gN_plummer(r, M, a)
    x    = gN / A0
    gobs = nu(x) * gN
    mu   = mu_from_x(x)
    S    = (2.0 * BETA / 3.0) * gN**2 / (mu**2 * C2)

    PsimPhi = poisson_solve(r, S)        # Psi - Phi (sign per Green's function above)
    Psi     = Psi_of_r(r, gobs)          # Psi(r) < 0
    # eta = Phi/Psi  with  Phi = Psi - (Psi-Phi); equivalently:
    eta_k   = 1.0 - PsimPhi / Psi
    eta_an  = eta_analytical(gobs)

    # well-sampled region: r_MOND < r < 0.2 * r_max_grid (avoid Psi boundary effect)
    mask = (r > rMOND) & (r < 0.2 * r_max_grid)
    resid = (eta_k[mask] - eta_an[mask]) / (eta_an[mask] - 1.0) * 100.0
    resid = resid[np.isfinite(resid)]
    max_abs = np.max(np.abs(resid)) if resid.size else float('inf')

    report = []
    report.append("KERNEL POINT-SOURCE TEST")
    report.append("="*60)
    report.append(f"M       = {M:.3e} Msun")
    report.append(f"r_eff   = {a} kpc (quasi-point)")
    report.append(f"r_MOND  = {rMOND:.4f} kpc")
    report.append(f"a0      = {A0:.3e} (km/s)^2/kpc")
    report.append(f"beta    = {BETA}")
    report.append("")
    report.append("Residual statistics for r > r_MOND:")
    report.append(f"  max|residual%| = {max_abs:.3f}%")
    report.append(f"  median |res%|  = {np.median(np.abs(resid)):.3f}%")
    report.append("")
    # sample points
    sample_idx = np.linspace(0, len(r)-1, 10).astype(int)
    report.append(f"{'r[kpc]':>10} {'g/a0':>10} {'eta_k-1':>12} {'eta_an-1':>12} {'res%':>10}")
    for i in sample_idx:
        report.append(f"{r[i]:10.3f} {gN[i]/A0:10.2e} {eta_k[i]-1:12.4e} {eta_an[i]-1:12.4e} "
                      f"{(eta_k[i]-eta_an[i])/(eta_an[i]-1)*100:10.2f}")
    report.append("")
    if max_abs < 10.0:
        verdict = "KERNEL CONFIRMED (max|res| < 10%)"
    else:
        # diagnose functional form: fit log(eta_k - 1) vs log(r) in deep-MOND
        deep = (gN/A0 < 0.1) & (r > rMOND) & (r < 0.2 * r_max_grid)
        if deep.sum() > 5:
            yk = np.abs(eta_k[deep]-1); ya = np.abs(eta_an[deep]-1)
            ok_k = np.isfinite(yk) & (yk > 0)
            ok_a = np.isfinite(ya) & (ya > 0)
            slope_k, _  = np.polyfit(np.log(r[deep][ok_k]), np.log(yk[ok_k]), 1)
            slope_a, _  = np.polyfit(np.log(r[deep][ok_a]), np.log(ya[ok_a]), 1)
            report.append(f"Functional form (deep-MOND log-log slope vs r):")
            report.append(f"  eta_kernel - 1   ~ r^{slope_k:.3f}")
            report.append(f"  eta_analytic - 1 ~ r^{slope_a:.3f}  (expected: +0.5)")
        verdict = f"KERNEL FAILED (max|res| = {max_abs:.1f}%) -- see functional form above"
    report.append(verdict)
    print("\n".join(report))

    with open(os.path.join(OUT, "kernel_pointsource_report.txt"), "w") as f:
        f.write("\n".join(report))

    # plot
    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    axes[0].loglog(r, gN, label='g_N')
    axes[0].loglog(r, gobs, label='g_obs')
    axes[0].axhline(A0, color='k', ls=':', label='a0')
    axes[0].axvline(rMOND, color='r', ls='--', label='r_MOND')
    axes[0].set_ylabel('acceleration [(km/s)^2/kpc]'); axes[0].legend(); axes[0].grid(True, which='both', alpha=0.3)

    axes[1].loglog(r, np.abs(eta_k-1), label='|eta_kernel - 1|')
    axes[1].loglog(r, np.abs(eta_an-1), '--', label='|eta_analytic - 1|')
    axes[1].axvline(rMOND, color='r', ls='--')
    axes[1].set_ylabel('|eta - 1|'); axes[1].legend(); axes[1].grid(True, which='both', alpha=0.3)

    axes[2].semilogx(r[mask], resid)
    axes[2].axhline(10, color='g', ls='--'); axes[2].axhline(-10, color='g', ls='--')
    axes[2].axvline(rMOND, color='r', ls='--')
    axes[2].set_ylabel('residual [%]'); axes[2].set_xlabel('r [kpc]'); axes[2].grid(True, which='both', alpha=0.3)
    axes[2].set_ylim(-200, 200)

    fig.suptitle("Paper III — Poisson kernel point-source validation")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kernel_pointsource_validation.png"), dpi=130)
    plt.close(fig)

    return max_abs < 10.0, dict(r=r, eta_k=eta_k, eta_an=eta_an, rMOND=rMOND)

# ------------------------------------------------------------------
# TASK 3: DGSAT-I
# ------------------------------------------------------------------
def run_dgsat():
    M    = 3.0e8
    a    = 2.1
    rMOND = np.sqrt(G * M / A0)
    eps   = a / rMOND
    print(f"\n[DGSAT-I] M={M:.2e}, r_eff={a} kpc, r_MOND={rMOND:.3f} kpc, eps={eps:.2f}")

    r = np.logspace(np.log10(0.01), np.log10(200.0), 4000)
    gN   = gN_plummer(r, M, a)
    x    = gN / A0
    gobs = nu(x) * gN
    mu   = mu_from_x(x)
    S    = (2.0 * BETA / 3.0) * gN**2 / (mu**2 * C2)

    PsimPhi = poisson_solve(r, S)
    Psi     = Psi_of_r(r, gobs)
    eta_k   = 1.0 - PsimPhi / Psi
    eta_an  = eta_analytical(gobs)

    # values at r_eff
    i_eff  = np.argmin(np.abs(r - a))
    i_MOND = np.argmin(np.abs(r - rMOND))
    eta_k_eff_pct  = (eta_k[i_eff]  - 1) * 100
    eta_an_eff_pct = (eta_an[i_eff] - 1) * 100
    factor = (eta_k[i_eff]-1) / (eta_an[i_eff]-1)

    lines = []
    lines.append("DGSAT-I KERNEL RESULT")
    lines.append("="*60)
    lines.append(f"M_total = {M:.2e} Msun")
    lines.append(f"r_eff   = {a} kpc")
    lines.append(f"r_MOND  = {rMOND:.4f} kpc")
    lines.append(f"epsilon = r_eff / r_MOND = {eps:.3f}")
    lines.append("")
    lines.append(f"At r = r_eff = {a} kpc:")
    lines.append(f"  eta_kernel - 1    = {eta_k_eff_pct:+.3f} %")
    lines.append(f"  eta_analytic - 1  = {eta_an_eff_pct:+.3f} %  (Paper I expectation ~9.9%)")
    lines.append(f"  correction factor = {factor:+.3f}")
    lines.append("")
    if factor > 1.0:
        interp = "factor > 1: Paper I is a LOWER LIMIT -- Euclid will see more."
    elif 0.1 < factor <= 1.0:
        interp = "factor in (0.1,1]: Paper I is an UPPER LIMIT -- still detectable."
    elif 0 < factor <= 0.1:
        interp = "factor in (0,0.1]: DGSAT-I is NOT a reliable target -- revise Paper I."
    else:
        interp = f"factor = {factor:+.3f} (sign-flipped): structural issue, see plot."
    lines.append(interp)
    print("\n".join(lines))

    with open(os.path.join(OUT, "kernel_dgsat_result.txt"), "w") as f:
        f.write("\n".join(lines))

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    axes[0].semilogx(r, (eta_k-1)*100, label='eta_kernel - 1')
    axes[0].semilogx(r, (eta_an-1)*100, '--', label='eta_analytic - 1 (Paper I)')
    axes[0].axvspan(0.5*rMOND, 50, color='gray', alpha=0.05)
    axes[0].axhspan(1, 5, color='g', alpha=0.15, label='Euclid sensitivity (1-5%)')
    axes[0].axvline(a,     color='b', ls=':', label=f'r_eff = {a} kpc')
    axes[0].axvline(rMOND, color='r', ls=':', label=f'r_MOND = {rMOND:.2f} kpc')
    axes[0].set_ylabel('eta - 1 [%]'); axes[0].legend(); axes[0].grid(True, which='both', alpha=0.3)
    axes[0].set_title(f"DGSAT-I — kernel vs Paper I (eps={eps:.2f})")

    axes[1].semilogx(r, (eta_k-1)/(eta_an-1))
    axes[1].axhline(1.0, color='k', ls=':')
    axes[1].axvline(a,     color='b', ls=':')
    axes[1].axvline(rMOND, color='r', ls=':')
    axes[1].set_ylabel('correction factor  eta_kernel / eta_analytic')
    axes[1].set_xlabel('r [kpc]'); axes[1].grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kernel_dgsat_result.png"), dpi=130)
    plt.close(fig)

    return factor

# ------------------------------------------------------------------
if __name__ == "__main__":
    ok, _ = run_pointsource()
    factor = run_dgsat()
    print("\n" + "="*60)
    print(f"Point-source kernel passed: {ok}")
    print(f"DGSAT-I correction factor : {factor:+.3f}")
