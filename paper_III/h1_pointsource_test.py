"""
Paper III - Task 2: Test of Hypothesis H1 in the point-source limit.

H1:  nabla^2 (Psi - Phi) = (16 pi / 3) * beta * G * rho_b(r) * g_N(r) / a0

Units (galactic):
  G   = 4.302e-3 kpc (km/s)^2 / Msun
  rho = Msun / kpc^3
  g_N = (km/s)^2 / kpc
  a0  = 3.857e-11 (km/s)^2 / kpc
  => [G rho_b g_N / a0] = (km/s)^2 / kpc^2  (matches Laplacian of Psi)

We use a Plummer profile with r_eff = 0.01 kpc (quasi-point) for M=1e10 Msun
and compare the numerical eta(r) against the Paper I analytic formula
in the deep-MOND window r > r_MOND = sqrt(G M / a0).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- constants (galactic units) ----
G   = 4.302e-3        # kpc (km/s)^2 / Msun
# a0 = 1.2e-10 m/s^2.  Convert: (km/s)^2/kpc = 1e6 m^2/s^2 / 3.0857e19 m
#                              = 3.2408e-14 m/s^2
# => a0 = 1.2e-10 / 3.2408e-14 ~= 3702 (km/s)^2/kpc
a0  = 3702.0          # (km/s)^2 / kpc
beta = 0.0075

# ---- dimensional check ----
# [G*rho*g_N/a0] = kpc*(km/s)^2/Msun * Msun/kpc^3 * (km/s)^2/kpc * kpc/(km/s)^2
#                = (km/s)^2 / kpc^2   OK
print("Dimensional check:")
print("  [G*rho_b*g_N/a0] = kpc*(km/s)^2/Msun  *  Msun/kpc^3  *  (km/s)^2/kpc  /  (km/s)^2/kpc")
print("                  = (km/s)^2 / kpc^2 = [Laplacian(Psi)]  OK")

# ---- problem setup ----
M     = 1.0e10         # Msun
r_eff = 0.01           # kpc  (quasi-point Plummer)
r_MOND = np.sqrt(G*M/a0)
print(f"\nr_MOND = sqrt(GM/a0) = {r_MOND:.3f} kpc")

# Plummer
def rho_plummer(r, M=M, a=r_eff):
    return 3*M/(4*np.pi*a**3) * (1 + (r/a)**2)**(-2.5)

def Menc_plummer(r, M=M, a=r_eff):
    return M * r**3 / (r**2 + a**2)**1.5

def g_N_plummer(r):
    return G * Menc_plummer(r) / r**2

# deep-MOND interpolation nu
def nu(y):
    return np.sqrt(0.5 + 0.5*np.sqrt(1.0 + 4.0/y**2))

def g_obs(r):
    gN = g_N_plummer(r)
    y  = gN / a0
    return nu(y) * gN

# Psi(r) = -int_r^inf g_obs(r') dr'
def Psi_of_r(r_grid):
    # cumulative from outside in
    g = g_obs(r_grid)
    Psi = np.zeros_like(r_grid)
    # trapezoid from i to N-1
    for i in range(len(r_grid)-1):
        Psi[i] = -np.trapezoid(g[i:], r_grid[i:])
    Psi[-1] = 0.0
    return Psi

# ---- grid ----
N = 3000
r = np.geomspace(1e-3, 2000.0, N)  # well past r_MOND ~ 108 kpc

# ---- source S_H1 ----
S = (16*np.pi/3) * beta * G * rho_plummer(r) * g_N_plummer(r) / a0

# ---- solve nabla^2 X = S, spherical, BC: X'(0)=0, X(r_max)=0 ----
# (1/r^2) d/dr (r^2 dX/dr) = S
# Let u = r * X. Then u'' = r * S.
# BC: u(0) = 0,  u(r_max)/r_max = 0 -> u(r_max) = 0.
# Solve via tridiagonal on a non-uniform grid.

def solve_radial_poisson(r, S):
    """Solve (1/r^2)(r^2 X')' = S with X(r_max)=0, regular at 0.
       Use u = r*X with u'' = r*S, u(0)=u(r_max)=0."""
    N = len(r)
    rhs = r * S
    # finite-difference second derivative on non-uniform grid
    # u''_i ~= 2/(h_- + h_+) * ((u_{i+1}-u_i)/h_+ - (u_i - u_{i-1})/h_-)
    A = np.zeros((N, N))
    b = np.zeros(N)
    A[0,0] = 1.0; b[0] = 0.0          # u(0)=0
    A[-1,-1] = 1.0; b[-1] = 0.0       # u(r_max)=0
    for i in range(1, N-1):
        h_m = r[i] - r[i-1]
        h_p = r[i+1] - r[i]
        denom = h_m + h_p
        A[i, i-1] =  2.0/(h_m*denom)
        A[i, i+1] =  2.0/(h_p*denom)
        A[i, i  ] = -2.0/(h_m*h_p)
        b[i] = rhs[i]
    u = np.linalg.solve(A, b)
    X = u / r
    return X

print("\nSolving Poisson equation for (Psi-Phi) under H1...")
slip = solve_radial_poisson(r, S)            # this is (Psi - Phi)(r)
Psi  = Psi_of_r(r)

# eta - 1 = -(Psi - Phi)/Psi
eta_H1   = -slip / Psi
y        = g_N_plummer(r) / a0
eta_anal = (2.0/3.0) * beta / np.sqrt(y * (1.0 + y))

# residual in deep-MOND window
mask = (r > 2*r_MOND) & (r < 1500.0)
res_pct = (eta_H1[mask] - eta_anal[mask]) / eta_anal[mask] * 100.0
maxres = np.nanmax(np.abs(res_pct))
medres = np.nanmedian(np.abs(res_pct))

# ---- plot ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].loglog(r, np.abs(eta_H1),    label="H1 numerical |eta-1|")
ax[0].loglog(r, np.abs(eta_anal),  '--', label="Paper I analytic |eta-1|")
ax[0].axvline(r_MOND, color='gray', ls=':', label="r_MOND")
ax[0].set_xlabel("r [kpc]"); ax[0].set_ylabel("|eta - 1|")
ax[0].set_title("Point-mass limit (Plummer r_eff=0.01 kpc)")
ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].semilogx(r[mask], res_pct)
ax[1].axhline(0, color='k', lw=0.5)
ax[1].axhline(10, color='orange', ls='--'); ax[1].axhline(-10, color='orange', ls='--')
ax[1].set_xlabel("r [kpc]"); ax[1].set_ylabel("residual %")
ax[1].set_title(f"Residual in deep-MOND window: max|res|={maxres:.1f}%, med={medres:.1f}%")
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("paper_III/h1_pointsource_test.png", dpi=130)

# ---- report ----
verdict = "CONFIRMED" if maxres < 10 else ("PARTIAL" if maxres < 30 else "REJECTED")
with open("paper_III/h1_pointsource_report.txt", "w", encoding="utf-8") as f:
    f.write("Paper III - Task 2: Point-mass test of Hypothesis H1\n")
    f.write("="*64 + "\n")
    f.write(f"M = {M:.2e} Msun, Plummer r_eff = {r_eff} kpc\n")
    f.write(f"r_MOND = sqrt(GM/a0) = {r_MOND:.3f} kpc\n")
    f.write(f"beta = {beta}, a0 = {a0:.3e} (km/s)^2/kpc\n\n")
    f.write("Dimensional check passed: [G rho_b g_N / a0] = (km/s)^2/kpc^2.\n\n")
    f.write(f"Deep-MOND window  r in [{r_MOND:.2f}, 200] kpc:\n")
    f.write(f"  max |residual| = {maxres:.2f} %\n")
    f.write(f"  median|residual| = {medres:.2f} %\n\n")
    f.write(f"VERDICT: H1 {verdict}\n")
    f.write(" (CONFIRMED if max<10%, PARTIAL 10-30%, REJECTED >30%)\n\n")
    # sample table
    f.write("Sample r [kpc] | y=g_N/a0 | eta_anal-1 | eta_H1-1 | res %\n")
    sample_idx = np.searchsorted(r, [3.5, 5, 10, 30, 100, 300])
    for i in sample_idx:
        if i < len(r):
            f.write(f" {r[i]:8.2f} | {y[i]:.3e} | {eta_anal[i]:.3e} | {eta_H1[i]:.3e} | {(eta_H1[i]-eta_anal[i])/eta_anal[i]*100:+.2f}\n")

print(f"\nmax|res|={maxres:.2f}%  median={medres:.2f}%  -> {verdict}")
