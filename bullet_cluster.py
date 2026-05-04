"""Bullet Cluster - DEV order-of-magnitude estimate."""
import numpy as np
import matplotlib.pyplot as plt

# Constants (SI)
G = 6.674e-11
Msun = 1.989e30
kpc = 3.0857e19
km = 1e3

# System parameters
M1 = 1.5e15 * Msun
M2 = 1.5e14 * Msun
M_tot = M1 + M2
v_rel = 3000 * km
f_gas = 0.15
r_c = 300 * kpc
r_c2 = 150 * kpc
d_sep = 720 * kpc

# DEV
beta = 0.0075
a0 = 1.2e-10

# ---- Task 1: characteristic acceleration
g_bullet = G * M_tot / r_c**2
ratio = g_bullet / a0

# ---- Task 2: relaxation time
# beta = gamma^2/(m_A^2 a0); A ~ (gamma/m_A^2) grad theta
# tau_A = 1/m_A. With L = sqrt(K)/m_A and K~1, m_A ~ 1/L.
# Use L ~ kpc scale (DEV stiffness length); estimate via galactic fits ~ a few kpc.
L_stiff = 5 * kpc  # typical DEV rigidity length from galactic regime
c = 3e8
m_A_inv_time = L_stiff / c  # tau_A in seconds (natural units c=1)
tau_A = m_A_inv_time
t_cross = r_c / v_rel
# In Myr
Myr = 3.156e13
tau_A_Myr = tau_A / Myr
t_cross_Myr = t_cross / Myr

# ---- Task 3: Hernquist subcluster, MOND interpolation
def g_newton_hernquist(r, M, a):
    return G * M / (r + a)**2

def nu_simple(y):
    # simple interpolating function: nu(y) = 0.5 + sqrt(0.25 + 1/y)
    return 0.5 + np.sqrt(0.25 + 1.0/y)

a_hern = r_c2  # scale
r = np.linspace(1*kpc, 1000*kpc, 500)
gN = g_newton_hernquist(r, M2, a_hern)
y = gN / a0
g_obs = nu_simple(y) * gN

# Lensing potential Psi(r) = -int_r^inf g_obs dr'  (so dPsi/dr = g_obs)
# Compute cumulative from outer edge
Psi = np.zeros_like(r)
for i in range(len(r)-2, -1, -1):
    Psi[i] = Psi[i+1] - g_obs[i] * (r[i+1]-r[i])

# eta(r): slip
g_for_eta = g_obs
x = g_for_eta / a0
eta_minus_1 = (2.0/3.0) * beta / np.sqrt(x*(1+x))
eta = 1 + eta_minus_1
Phi = eta * Psi

# Centroid separation estimate:
# Gas decelerated by ram pressure: gas lags by Delta_gas = f_decel * v_rel * t_merger
# Use t_merger ~ d_sep / v_rel as merger duration since pericenter
t_merger = d_sep / v_rel

def delta_x(f_decel):
    # gas lag: gas slowed by f_decel*v_rel over t_merger
    # displacement difference between DM (no drag) and gas
    return f_decel * v_rel * t_merger

dx_central = delta_x(0.3) / kpc
dx_low = delta_x(0.1) / kpc
dx_high = delta_x(0.5) / kpc

# ---- Task 5: Plots
fig, ax = plt.subplots(figsize=(7,5))
ax.loglog(r/kpc, gN, label=r'$g_N(r)$ Newtonian')
ax.loglog(r/kpc, g_obs, label=r'$g_{\rm obs}(r)$ DEV')
ax.axvline(r_c2/kpc, color='gray', ls='--', label=r'$r_c$')
ax.axhline(a0, color='red', ls=':', label=r'$a_0$')
ax.set_xlabel('r [kpc]'); ax.set_ylabel('g [m/s$^2$]')
ax.set_title('Bullet Cluster subcluster acceleration')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('bullet_cluster_acceleration.png', dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(7,5))
ax.semilogx(x, eta_minus_1)
ax.axvspan(1, 1e3, color='green', alpha=0.2, label='Bullet Cluster regime')
ax.set_xlabel(r'$g/a_0$'); ax.set_ylabel(r'$\eta - 1$')
ax.set_title('DEV gravitational slip')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('bullet_cluster_slip.png', dpi=120)
plt.close()

# ---- Task 4: results
with open('bullet_results.txt','w') as f:
    f.write("=== Bullet Cluster DEV Analysis ===\n\n")
    f.write(f"M_tot = {M_tot/Msun:.2e} Msun\n")
    f.write(f"r_c = {r_c/kpc:.0f} kpc\n")
    f.write(f"v_rel = {v_rel/km:.0f} km/s\n\n")
    f.write(f"g_bullet = {g_bullet:.3e} m/s^2\n")
    f.write(f"a0 = {a0:.2e} m/s^2\n")
    f.write(f"g_bullet / a0 = {ratio:.2f}\n")
    if ratio > 10:
        f.write("==> Regime NEWTONIAN (mu -> 1); DEV slip suppressed\n\n")
    elif ratio > 0.3:
        f.write("==> Regime de TRANSICAO\n\n")
    else:
        f.write("==> Regime MOND profundo\n\n")
    f.write(f"tau_A (rigidity timescale) ~ {tau_A_Myr:.3e} Myr\n")
    f.write(f"t_cross = r_c/v_rel = {t_cross_Myr:.2f} Myr\n")
    f.write(f"tau_A / t_cross = {tau_A/t_cross:.3e}\n")
    f.write("(tau_A << t_cross: A_mu rastreia theta quase instantaneamente)\n\n")
    eta_at_rc = np.interp(r_c2, r, eta_minus_1)
    f.write(f"eta-1 at r_c2 = {eta_at_rc:.4e}\n\n")
    f.write(f"--- Separation estimates (ram-pressure lag model) ---\n")
    f.write(f"f_decel=0.3: Delta_x = {dx_central:.0f} kpc\n")
    f.write(f"f_decel=0.1: Delta_x = {dx_low:.0f} kpc\n")
    f.write(f"f_decel=0.5: Delta_x = {dx_high:.0f} kpc\n")
    f.write(f"Observed: ~200 kpc\n\n")
    if 100 <= dx_central <= 400:
        f.write("==> CONSISTENT with Bullet Cluster observation\n")
    elif dx_central < 100:
        f.write("==> Underpredicts separation\n")
    else:
        f.write("==> Overpredicts separation\n")

print(open('bullet_results.txt').read())

# ---- Task 6: LaTeX
latex = r"""\section*{Appendix C: Bullet Cluster Order-of-Magnitude Estimate}

We assess whether the DEV theory is qualitatively consistent with the
$\sim 200$~kpc lensing--gas centroid offset observed in 1E~0657-558.
At the core radius $r_c \approx 300$~kpc, the characteristic Newtonian
acceleration is $g_{\rm bullet} = G M_{\rm tot}/r_c^2 \approx %.2e$~m\,s$^{-2}$,
giving $g_{\rm bullet}/a_0 \approx %.0f$. The Bullet Cluster therefore
operates deep in the Newtonian regime ($\mu \to 1$), where the DEV
modification of gravity is suppressed and the gravitational slip
$\eta - 1 \sim (2\beta/3)/\sqrt{x(1+x)}$ becomes small ($\sim %.1e$ at $r_c$).
A simple ram-pressure lag model in which the gas is decelerated by
a fraction $f_{\rm decel} \in [0.1, 0.5]$ of $v_{\rm rel} \approx 3000$~km\,s$^{-1}$
over the merger crossing time predicts a centroid separation
$\Delta x \in [%.0f, %.0f]$~kpc, bracketing the observed value.
We emphasize that this is an order-of-magnitude estimate; the actual
separation in DEV is governed by the inertia and rigidity of the vector
field $A_\mu$ during the merger, and a quantitative prediction requires
full numerical simulation, which we leave to future work.
""" % (g_bullet, ratio, eta_at_rc, dx_low, dx_high)

with open('bullet_appendix.tex','w') as f:
    f.write(latex)
print("\nFiles written: bullet_results.txt, bullet_cluster_acceleration.png, bullet_cluster_slip.png, bullet_appendix.tex")
