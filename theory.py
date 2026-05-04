"""
DEV Theory Module
=================
"""

import numpy as np

# Constantes físicas
A0 = 1.2e-10                    # m/s^2, escala MOND
X0 = 0.5 * A0**2
G_NEWTON = 6.674e-11            # m^3 kg^-1 s^-2
KPC_TO_M = 3.0857e19            # m/kpc
MSUN = 1.989e30                 # kg

# Valor canônico de alpha derivado analiticamente
ALPHA_SPHERICAL = 2.0/3.0
ALPHA_DISK      = 1.0
ALPHA_OBLATE    = 0.85


def mu_dev(x):
    x = np.asarray(x, dtype=float)
    return x / np.sqrt(1.0 + x**2)


def nu_dev(y):
    y = np.asarray(y, dtype=float)
    return np.sqrt(0.5 + 0.5 * np.sqrt(1.0 + 4.0 / y**2))


def v_circ_dev(r_kpc, M_bar_enclosed_msun):
    r_m = np.asarray(r_kpc) * KPC_TO_M
    M_kg = np.asarray(M_bar_enclosed_msun) * MSUN
    g_N = G_NEWTON * M_kg / r_m**2
    y = g_N / A0
    g_obs = nu_dev(y) * g_N
    v_m_s = np.sqrt(g_obs * r_m)
    return v_m_s / 1000.0


def eta_dev(g_obs, beta, alpha=None, geometry='spherical'):
    """eta(g) - 1 = alpha * beta / sqrt(x*(1+x)),  x = g/a0"""
    if alpha is None:
        alpha_map = {
            'spherical': ALPHA_SPHERICAL,
            'oblate':    ALPHA_OBLATE,
            'disk':      ALPHA_DISK,
        }
        alpha = alpha_map.get(geometry, ALPHA_SPHERICAL)
    x = np.asarray(g_obs) / A0
    f = 1.0 / np.sqrt(x * (1.0 + x))
    return 1.0 + alpha * beta * f


def regime(g_obs):
    x = np.asarray(g_obs) / A0
    if np.isscalar(x) or x.ndim == 0:
        if x > 10:    return "Newton (vácuo inerte)"
        elif x < 0.1: return "MOND profundo (vácuo saturado)"
        else:         return "Transição"
    return np.where(x > 10, "Newton",
            np.where(x < 0.1, "MOND profundo", "Transição"))


if __name__ == "__main__":
    print("alpha_spherical =", ALPHA_SPHERICAL)
    for x in [100, 10, 1.0, 0.1, 0.01, 0.001]:
        g = x * A0
        e = float(eta_dev(g, beta=0.01)) - 1
        print(f"  x={x:8.3f}  eta-1={e:.5f}")
