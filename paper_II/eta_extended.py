"""
Paper II - extended-source gravitational slip via Green's function convolution.
Solves  d^2 f/dr^2 + (2/r) df/dr = S(r),  f = Psi - Phi,
with  S(r) = (2 beta / 3) * g_N(r)^2 / c^2,
in galactic units: kpc, km/s, Msun.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Constants in galactic units
G   = 4.302e-6           # kpc (km/s)^2 / Msun
a0  = 3.703e3            # (km/s)^2 / kpc   (1.2e-10 m/s^2 * 3.086e13)
c   = 2.998e5            # km/s
c2  = c*c                # (km/s)^2
beta = 0.0075

def nu(y):
    return np.sqrt(0.5 + 0.5*np.sqrt(1.0 + 4.0/np.maximum(y, 1e-30)**2))

# -------- Mass profiles --------
def hernquist(r):
    M_tot, r_s = 1.0e11, 5.0
    M = M_tot * r**2 / (r + r_s)**2
    gN = G * M / r**2
    label = "Hernquist (M=1e11, r_s=5 kpc)"
    rscale = r_s
    return M, gN, M_tot, rscale, label

def plummer(r):
    M_tot, r_eff = 3.0e8, 4.7   # DGSAT-I
    M = M_tot * r**3 / (r**2 + r_eff**2)**1.5
    gN = G * M / r**2
    label = "Plummer DGSAT-I (M=3e8, r_eff=4.7 kpc)"
    rscale = r_eff
    return M, gN, M_tot, rscale, label

def nfw(r):
    r_s, r_max_t = 10.0, 200.0
    # find rho_0 such that M(10 kpc) = 1e10 Msun
    def M_of(rho0, rr):
        x = rr/r_s
        return 4*np.pi*rho0*r_s**3 * (np.log(1+x) - x/(1+x))
    target = 1.0e10
    x10 = 10.0/r_s
    rho0 = target / (4*np.pi*r_s**3 * (np.log(1+x10) - x10/(1+x10)))
    M = np.where(r < r_max_t, M_of(rho0, r), M_of(rho0, r_max_t))
    gN = G * M / r**2
    label = "NFW (rho0 fit M(10)=1e10, r_s=10 kpc)"
    rscale = r_s
    return M, gN, M_of(rho0, r_max_t), rscale, label

# -------- Solver --------
def solve_profile(profile_fn, name):
    r_min, r_max, N = 0.01, 500.0, 3000
    r = np.linspace(r_min, r_max, N)
    dr = r[1] - r[0]
    M, gN, M_tot, rscale, label = profile_fn(r)
    y = gN / a0
    g_obs = nu(y) * gN
    # Psi(r) = -int_r^{rmax} g_obs dr'  (so Psi -> 0 at rmax, negative inward)
    Psi = -cumulative_trapezoid(g_obs[::-1], r[::-1], initial=0.0)[::-1]
    # cumtrapz from rmax inward; Psi(rmax)=0, Psi negative for r<rmax
    # Actually: trapezoid of g_obs from r to rmax, with sign:
    Psi = -np.flip(cumulative_trapezoid(np.flip(g_obs), -np.flip(r), initial=0.0))
    # Source S(r) = (2 beta/3) g_N^2 / c^2
    S = (2.0/3.0) * beta * gN**2 / c2

    # Tridiagonal solve of f'' + (2/r) f' = S, f(rmax)=0, f'(rmin)=0 (Neumann)
    A = np.zeros((N, N))
    b = np.zeros(N)
    # Interior: i=1..N-2
    for i in range(1, N-1):
        ri = r[i]
        A[i, i-1] = 1.0/dr**2 - 1.0/(ri*dr)
        A[i, i  ] = -2.0/dr**2
        A[i, i+1] = 1.0/dr**2 + 1.0/(ri*dr)
        b[i] = S[i]
    # Neumann at i=0:  f1 = f0  -> A[0,0]=-1, A[0,1]=1, b=0
    A[0, 0] = -1.0; A[0, 1] = 1.0; b[0] = 0.0
    # Dirichlet at i=N-1: f=0
    A[N-1, N-1] = 1.0; b[N-1] = 0.0
    # Use banded solver via scipy
    from scipy.linalg import solve_banded
    ab = np.zeros((3, N))
    ab[0, 1:] = np.diag(A, k=1)
    ab[1, :]  = np.diag(A, k=0)
    ab[2, :-1]= np.diag(A, k=-1)
    f = solve_banded((1,1), ab, b)

    # eta - 1
    # Note: Psi is negative (potential well). Define potentials with Psi<0.
    # eta - 1 = -(Psi - Phi)/Psi = -f/Psi
    eta_num_m1 = -f / Psi
    yobs = g_obs / a0
    eta_an_m1 = (2.0/3.0) * beta / np.sqrt(yobs * (1.0 + yobs))

    # r_MOND
    r_MOND = np.sqrt(G * M_tot / a0)

    return dict(name=name, label=label, r=r, gN=gN, g_obs=g_obs,
                Psi=Psi, f=f, eta_num=eta_num_m1, eta_an=eta_an_m1,
                r_MOND=r_MOND, rscale=rscale, M_tot=M_tot)

results = {p[0]: solve_profile(p[1], p[0]) for p in
           [("Hernquist", hernquist), ("Plummer", plummer), ("NFW", nfw)]}

# -------- Validity analysis --------
def regimes(d):
    r = d['r']; rM = d['r_MOND']
    res = (d['eta_num'] - d['eta_an']) / np.where(np.abs(d['eta_an'])>1e-30, d['eta_an'], 1e-30) * 100.0
    out = {}
    for tag, mask in [("r>rMOND", r > rM),
                      ("r<rMOND", r < rM),
                      ("r>3rMOND", r > 3*rM),
                      ("r=r_eff", np.argmin(np.abs(r - d['rscale']))),
                      ("r=rMOND", np.argmin(np.abs(r - rM)))]:
        if isinstance(mask, np.ndarray):
            mm = mask & np.isfinite(res)
            if mm.any():
                out[tag] = (np.nanmean(np.abs(res[mm])), np.nanmax(np.abs(res[mm])))
            else:
                out[tag] = (np.nan, np.nan)
        else:
            out[tag] = (abs(res[mask]), abs(res[mask]))
    return res, out

reports = {k: regimes(v) for k, v in results.items()}

# -------- Figures --------
plt.rcParams.update({'font.size': 10})

# Figure 1: 3-panel comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, key in zip(axes, results.keys()):
    d = results[key]
    ax.loglog(d['r'], np.abs(d['eta_num']), 'b-', label=r'$\eta_{\rm num}-1$')
    ax.loglog(d['r'], np.abs(d['eta_an']),  'r--', label=r'$\eta_{\rm anal}-1$')
    ax.axvline(d['r_MOND'], color='k', ls=':', alpha=0.6, label=r'$r_{\rm MOND}$')
    ax.axhspan(0.01, 0.05, color='green', alpha=0.15, label='Euclid 1-5%')
    ax.set_title(d['label'], fontsize=9)
    ax.set_xlabel('r [kpc]'); ax.set_ylabel(r'$|\eta-1|$')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('eta_extended_profiles.png', dpi=130)
plt.close()

# Figure 2: residuals
fig, ax = plt.subplots(figsize=(7,5))
for key, color in [("Hernquist","C0"), ("Plummer","C1"), ("NFW","C2")]:
    d = results[key]; res, _ = reports[key]
    ax.semilogx(d['r']/d['r_MOND'], res, color=color, label=key)
ax.axhline(10, ls='--', color='gray'); ax.axhline(-10, ls='--', color='gray')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel(r'$r/r_{\rm MOND}$'); ax.set_ylabel('residual [%]')
ax.set_ylim(-60, 60); ax.legend(); ax.grid(alpha=0.3)
ax.set_title(r'$(\eta_{\rm num}-\eta_{\rm anal})/\eta_{\rm anal}$')
plt.tight_layout(); plt.savefig('eta_residuals.png', dpi=130); plt.close()

# Figure 3: DGSAT-I focus
d = results['Plummer']
fig, ax = plt.subplots(figsize=(7,5))
ax.loglog(d['r'], np.abs(d['eta_num']), 'b-', lw=2, label=r'$\eta_{\rm num}-1$')
ax.loglog(d['r'], np.abs(d['eta_an']),  'r--', lw=2, label=r'$\eta_{\rm anal}-1$')
ax.axvline(d['rscale'], color='C2', ls=':', label=r'$r_{\rm eff}=4.7$ kpc')
ax.axvline(3*d['r_MOND'], color='C3', ls=':', label=r'$3 r_{\rm MOND}$')
ax.axhspan(0.01, 0.05, color='green', alpha=0.15)
ax.set_xlabel('r [kpc]'); ax.set_ylabel(r'$|\eta-1|$')
ax.set_title('DGSAT-I (Plummer): extended vs point-source slip')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('eta_dgsat_profile.png', dpi=130); plt.close()

# Figure 4: validity table (rendered as figure)
fig, ax = plt.subplots(figsize=(8, 2.5)); ax.axis('off')
rows = []
for key in ["Hernquist", "Plummer", "NFW"]:
    d = results[key]; _, rg = reports[key]
    res_at_eff = rg['r=r_eff']
    res_far = rg['r>3rMOND']
    valid = "YES" if res_far[1] < 10 else ("MARGINAL" if res_far[1] < 30 else "NO")
    rows.append([key, f"{d['r_MOND']:.2f}", f"{d['rscale']/d['r_MOND']:.3f}",
                 f"{res_at_eff[0]:.1f}%", f"{res_far[1]:.1f}%", valid])
table = ax.table(cellText=rows,
    colLabels=["Profile","r_MOND [kpc]","r_scale/r_MOND","|res| @ r_scale","max|res| r>3rMOND","Valid?"],
    loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.6)
plt.savefig('validity_table.png', dpi=130, bbox_inches='tight'); plt.close()

# -------- Text dump --------
with open("eta_extended_results.txt", "w") as fout:
    fout.write("Paper II - eta extended-source convolution results\n")
    fout.write(f"beta = {beta},  a0 = {a0:.3e} (km/s)^2/kpc,  c^2 = {c2:.3e}\n\n")
    for key in ["Hernquist", "Plummer", "NFW"]:
        d = results[key]; _, rg = reports[key]
        fout.write(f"=== {d['label']} ===\n")
        fout.write(f"  M_tot       = {d['M_tot']:.3e} Msun\n")
        fout.write(f"  r_scale     = {d['rscale']:.3f} kpc\n")
        fout.write(f"  r_MOND      = {d['r_MOND']:.3f} kpc\n")
        fout.write(f"  r_scale/r_MOND = {d['rscale']/d['r_MOND']:.4f}\n")
        for tag, val in rg.items():
            fout.write(f"  |res| {tag:10s}: mean={val[0]:.2f}%  max={val[1]:.2f}%\n")
        fout.write("\n")
print("Done.")
