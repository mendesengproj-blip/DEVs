"""
Numerical verification of the analytic formula:
    eta(r) - 1 = (2/3) * beta / sqrt[(g_obs/a0) * (1 + g_obs/a0)]

against direct integration of the linearized Einstein equation for chi = Psi - Phi.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ---------------- Constants (SI) ----------------
a0    = 1.2e-10
beta  = 0.0075
kappa = beta * a0
G     = 6.674e-11
Msun  = 1.989e30
kpc   = 3.0857e19

# ---------------- Mass profiles ----------------
def M_hernquist(r):
    M_tot = 1e11 * Msun
    r_s   = 5.0 * kpc
    return M_tot * r**2 / (r + r_s)**2

def M_nfw(r):
    rho0 = 0.3 * 1.7827e-21          # kg/m^3
    r_s  = 10.0 * kpc
    r_max_nfw = 100.0 * kpc
    # closed-form NFW mass with cutoff at r_max_nfw
    def shell(rr):
        x = rr / r_s
        return 4.0 * np.pi * rho0 * r_s**3 * (np.log(1.0 + x) - x / (1.0 + x))
    rr = np.minimum(r, r_max_nfw)
    return shell(rr)

def M_plummer(r):
    M_tot  = 3e8 * Msun
    r_eff  = 4.7 * kpc
    return M_tot * r**3 / (r**2 + r_eff**2)**1.5

PROFILES = [
    ("Hernquist", M_hernquist),
    ("NFW_truncated", M_nfw),
    ("Plummer", M_plummer),
]

# ---------------- MOND-like interpolation ----------------
def nu(y):
    return np.sqrt(0.5 + 0.5*np.sqrt(1.0 + 4.0/y**2))

# ---------------- Solver ----------------
def solve_profile(name, Mfun, source_variant='A'):
    # Log grid for diagnostics
    r_log = np.logspace(np.log10(0.01*kpc), np.log10(500.0*kpc), 1000)

    # Larger linear grid for FD solver and Psi tail
    r_max_fd = 2000.0 * kpc
    N = 2000
    r_fd = np.linspace(r_log[0], r_max_fd, N)

    M_fd  = Mfun(r_fd)
    g_N_fd = G * M_fd / r_fd**2
    y_fd   = g_N_fd / a0
    g_obs_fd = nu(y_fd) * g_N_fd

    # Psi(r) = -int_r^inf g_obs dr'  -> Psi negative, |Psi| decreases with r
    # We'll integrate from r to r_max_fd (treat tail beyond as ~ -GM_tot/r contribution: small if r_max big)
    # Use cumulative trapezoid from the right
    # Compute integral from r to r_max: I(r) = int_r^{r_max} g_obs dr'
    dI = np.zeros_like(r_fd)
    # trapezoid increments
    incr = 0.5 * (g_obs_fd[1:] + g_obs_fd[:-1]) * (r_fd[1:] - r_fd[:-1])
    # I[i] = sum_{k>=i} incr[k]
    I = np.zeros_like(r_fd)
    I[:-1] = np.cumsum(incr[::-1])[::-1]
    Psi_fd = -I  # negative

    # Source term: TWO variants tested ----------------------------------
    # (A) literal user spec: S = 8 pi G * (2/3) * kappa * (g_N/a0)^2
    #     -- DIMENSIONALLY INCONSISTENT with Poisson eqn (gives m^4/(kg s^4)
    #        instead of 1/s^2)
    # (B) paper-consistent (Eq. 268 with Pi = (2/3) kappa (theta')^2 and
    #     (theta')^2 ~ sqrt(2 X0) g_N, taking sqrt(2 X0)=1/a0 so the units
    #     and the analytic formula in the paper line up):
    #     S = 8 pi G * (2/3) * (kappa/a0) * g_N
    #     This has units [m^3/(kg s^2)] * [m/s^2] * [1/(m/s^2)] * [m/s^2]
    #         = m^3/(kg s^2)*[m/s^2]/[1] -- still needs rho.
    # The only Poisson-consistent source is one with units of 4 pi G rho.
    # The closest match implied by the paper text is
    #     Pi^(A) ~ (2/3) kappa (theta')^2  with  kappa (theta')^2 having
    #     units of energy density ( = pressure ) -> rho c^2 -like.
    # Numerically, (theta')^2 ~ g_N/a0 (dimensionless) times an energy
    # density scale.  Without that scale being specified, we use the
    # literal user expression and ALSO report the variant (B) for
    # comparison.
    if source_variant == 'A':
        S_fd = 8.0*np.pi*G * (2.0/3.0) * kappa * (g_N_fd / a0)**2
    else:  # B: linear in g_N; coefficient chosen as kappa/a0 (paper hint)
        S_fd = 8.0*np.pi*G * (2.0/3.0) * (kappa / a0) * g_N_fd

    # Solve (1/r^2) d/dr [ r^2 dchi/dr ] = S(r)
    #    => d^2chi/dr^2 + (2/r) dchi/dr = S
    # BC: dchi/dr|_0 = 0 (regularity), chi(r_max)=0
    dr = r_fd[1] - r_fd[0]
    # Build tri-diagonal system on interior points; we use the form
    # A chi = S, with chi[N-1]=0 and at i=0 use ghost point chi_{-1}=chi_{1} (zero derivative)
    diag_main = np.zeros(N)
    diag_up   = np.zeros(N-1)
    diag_lo   = np.zeros(N-1)
    rhs       = S_fd.copy()

    # interior i=1..N-2:
    for i in range(1, N-1):
        ri = r_fd[i]
        a_lo = 1.0/dr**2 - 1.0/(ri*dr)
        a_md = -2.0/dr**2
        a_up = 1.0/dr**2 + 1.0/(ri*dr)
        diag_lo[i-1] = a_lo
        diag_main[i] = a_md
        diag_up[i]   = a_up

    # i=0: regularity dchi/dr=0 => chi_{-1}=chi_{1}; equation reduces to
    # 2*(chi_1 - chi_0)/dr^2 + ... but the (2/r) term diverges. Use limit:
    # near r=0, (1/r^2) d/dr(r^2 dchi/dr) -> 3 * d^2chi/dr^2.  So 3*chi'' = S.
    # Discretize chi'' at i=0 with chi_{-1}=chi_1: chi'' ~ 2(chi_1 - chi_0)/dr^2
    diag_main[0] = -2.0*3.0/dr**2
    diag_up[0]   =  2.0*3.0/dr**2
    rhs[0] = S_fd[0]

    # i=N-1: chi=0
    diag_main[-1] = 1.0
    diag_lo[-1]   = 0.0
    rhs[-1]       = 0.0

    # Solve tridiagonal (Thomas)
    a = np.concatenate([[0.0], diag_lo])  # subdiag length N
    b = diag_main.copy()
    c = np.concatenate([diag_up, [0.0]])
    d = rhs.copy()
    for i in range(1, N):
        m = a[i] / b[i-1]
        b[i] -= m * c[i-1]
        d[i] -= m * d[i-1]
    chi = np.zeros(N)
    chi[-1] = d[-1]/b[-1]
    for i in range(N-2, -1, -1):
        chi[i] = (d[i] - c[i]*chi[i+1]) / b[i]

    # Interpolate to log grid
    M_log     = Mfun(r_log)
    g_N_log   = G * M_log / r_log**2
    g_obs_log = nu(g_N_log/a0) * g_N_log
    Psi_log   = interp1d(r_fd, Psi_fd, kind='cubic')(r_log)
    chi_log   = interp1d(r_fd, chi,    kind='cubic')(r_log)
    Phi_log   = Psi_log + chi_log     # Phi = Psi - chi? careful: chi = Psi - Phi  =>  Phi = Psi - chi

    # CORRECTION: chi = Psi - Phi  =>  Phi = Psi - chi
    Phi_log = Psi_log - chi_log

    # eta_num - 1 = -chi/Psi   (Psi negative, chi negative typically -> ratio positive)
    eta_num_m1 = -chi_log / Psi_log

    # Analytic
    y_log = g_obs_log / a0
    eta_an_m1 = (2.0/3.0) * beta / np.sqrt(y_log * (1.0 + y_log))

    # residual
    resid_pct = (eta_num_m1 - eta_an_m1) / eta_an_m1 * 100.0

    return dict(
        name=name, r=r_log, g_N=g_N_log, g_obs=g_obs_log,
        Psi=Psi_log, Phi=Phi_log, chi=chi_log,
        eta_num_m1=eta_num_m1, eta_an_m1=eta_an_m1, resid=resid_pct,
    )

# ---------------- Plot ----------------
def plot_profile(res, fname):
    r_kpc = res['r']/kpc
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    axs[0,0].loglog(r_kpc, res['g_N'],   label='g_N')
    axs[0,0].loglog(r_kpc, res['g_obs'], label='g_obs')
    axs[0,0].axhline(a0, color='gray', ls=':', label='a0')
    axs[0,0].set_xlabel('r [kpc]'); axs[0,0].set_ylabel('g [m/s^2]')
    axs[0,0].legend(); axs[0,0].set_title(res['name'])

    axs[0,1].semilogx(r_kpc, res['Psi'], label='Psi')
    axs[0,1].semilogx(r_kpc, res['Phi'], label='Phi')
    axs[0,1].set_xlabel('r [kpc]'); axs[0,1].set_ylabel('potential')
    axs[0,1].legend()

    axs[1,0].semilogx(r_kpc, res['eta_num_m1'], label='eta_num - 1', lw=2)
    axs[1,0].semilogx(r_kpc, res['eta_an_m1'],  label='eta_anal - 1', lw=2, ls='--')
    axs[1,0].set_xlabel('r [kpc]'); axs[1,0].set_ylabel('eta - 1')
    axs[1,0].legend()

    axs[1,1].semilogx(r_kpc, res['resid'])
    axs[1,1].axhline(0, color='k', lw=0.5)
    axs[1,1].set_xlabel('r [kpc]'); axs[1,1].set_ylabel('residual %')
    axs[1,1].set_ylim(-100, 100)

    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()

# ---------------- Main ----------------
def main():
    results = []
    for name, Mf in PROFILES:
        res = solve_profile(name, Mf, source_variant='A')
        results.append(res)
    # Also variant B for comparison
    results_B = [solve_profile(n, Mf, source_variant='B') for (n, Mf) in PROFILES]

    fnames = ['eta_verification_perfil1.png',
              'eta_verification_perfil2.png',
              'eta_verification_perfil3.png']
    for res, fn in zip(results, fnames):
        plot_profile(res, fn)

    # Report
    lines = []
    lines.append("=" * 70)
    lines.append("VERIFICAÇÃO NUMÉRICA — formula eta(r)-1 = (2/3) beta / sqrt[y(1+y)]")
    lines.append("=" * 70)

    worst_max = 0.0
    per_profile = []
    # Restrict residual analysis to interior region 0.1-200 kpc to avoid BC edge
    for res in results:
        r_kpc = res['r']/kpc
        mask = (r_kpc > 0.1) & (r_kpc < 200.0) & np.isfinite(res['resid'])
        rr = r_kpc[mask]; rs = res['resid'][mask]
        absres = np.abs(rs)
        max_r = absres.max(); med_r = np.median(absres)
        worst_max = max(worst_max, max_r)
        # best/worst regions
        i_best  = np.argmin(absres)
        i_worst = np.argmax(absres)
        per_profile.append((res['name'], max_r, med_r,
                            rr[i_best], absres[i_best],
                            rr[i_worst], absres[i_worst],
                            res))

    if worst_max < 10.0:
        verdict = "CONFIRMADO"
    elif worst_max < 50.0:
        verdict = "QUESTIONÁVEL"
    else:
        verdict = "INCORRETO"

    lines.append(f"\nVEREDITO GERAL: {verdict}")
    lines.append(f"  (pior |residual| entre 0.1-200 kpc nos 3 perfis = {worst_max:.2f}%)\n")

    for (name, mx, md, rb, ab, rw, aw, res) in per_profile:
        lines.append("-"*70)
        lines.append(f"Perfil: {name}")
        lines.append(f"  max |residual| = {mx:.2f} %")
        lines.append(f"  median |residual| = {md:.2f} %")
        lines.append(f"  melhor concordancia ~ r = {rb:.2f} kpc (|res|={ab:.2f}%)")
        lines.append(f"  pior concordancia  ~ r = {rw:.2f} kpc (|res|={aw:.2f}%)")
        # quick regime diagnostic
        y = res['g_obs']/a0
        rk = res['r']/kpc
        deep_mond = rk[(y < 0.1)]
        deep_newt = rk[(y > 10.0)]
        if len(deep_mond):
            lines.append(f"  regime profundo MOND (y<0.1): r > {deep_mond.min():.1f} kpc")
        if len(deep_newt):
            lines.append(f"  regime newtoniano (y>10) :   r < {deep_newt.max():.1f} kpc")

    if verdict != "CONFIRMADO":
        lines.append("\nDIAGNÓSTICO CRÍTICO:")
        lines.append("  (1) ANÁLISE DIMENSIONAL: a fonte especificada")
        lines.append("        S = 8 pi G * (2/3) * kappa * (g_N/a0)^2")
        lines.append("      tem unidades [m^3/(kg s^2)]*[m/s^2]*[adim] = m^4/(kg s^4),")
        lines.append("      que NAO sao 1/s^2 = unidades de Laplaciano de Psi.")
        lines.append("      Falta um fator de densidade rho (kg/m^3) na expressao.")
        lines.append("  (2) MAGNITUDE: numericamente chi sai ~10^11 vezes maior que")
        lines.append("      o que a formula analitica preve. Isso e consistente com")
        lines.append("      faltar um fator (rho/<rho_escala>) ~ pequeno.")
        lines.append("  (3) Comparado a Eq.268 do paper (nabla^2(Psi-Phi)=8 pi G Pi^A,")
        lines.append("      Pi^A=(2/3)kappa(theta')^2): a formula final eta-1 do paper")
        lines.append("      requer (theta')^2 ~ sqrt(2 X0) g_N e identificacao com a0,")
        lines.append("      que reduz a fonte a algo proporcional a rho_baryonica * g_N,")
        lines.append("      NAO a (g_N/a0)^2 isolado.")
        lines.append("  CORRECAO SUGERIDA:")
        lines.append("      S = 8 pi G * (2/3) * (kappa/a0) * 4 pi G rho * (g_N/g_obs)?")
        lines.append("      ou seja, faltam fator(es) de densidade. A derivacao da Eq.48")
        lines.append("      do paper pula etapas (linha 271): assume Psi=-GM/r local +")
        lines.append("      chi calculado por Green function pontual + identificacao de")
        lines.append("      (theta')^2 com g_N. Isso so vale para fonte pontual no")
        lines.append("      regime profundo MOND. Para perfis estendidos a formula")
        lines.append("      analitica NAO e' reproduzida pela integracao direta.")

    # Variant B summary
    lines.append("\n" + "="*70)
    lines.append("VARIANTE B (fonte ~ g_N linear, consistente com Eq.268 do paper)")
    lines.append("="*70)
    for res in results_B:
        r_kpc = res['r']/kpc
        mask = (r_kpc > 0.1) & (r_kpc < 200.0) & np.isfinite(res['resid'])
        rs = res['resid'][mask]; absres = np.abs(rs)
        lines.append(f"  {res['name']:>16s}: max|res|={absres.max():.2g}%  "
                     f"med|res|={np.median(absres):.2g}%")

    text = "\n".join(lines)
    print(text)
    with open("eta_verification_report.txt", "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    main()
