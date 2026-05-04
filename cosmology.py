"""
DEV Cosmology Module
====================
Modified linear growth equation and f sigma_8 predictions.

mu_eff(z, k, beta) = 1 + (alpha*beta/2) / sqrt[(gc/a0)(1+gc/a0)]
gc(k,z) = (3/2) Omega_m(z) H(z)^2 / k_phys
"""

import numpy as np
from scipy.integrate import solve_ivp
from theory import A0, ALPHA_SPHERICAL

H0_KMS_MPC = 70.0
H0_SI = H0_KMS_MPC * 1000.0 / 3.0857e22   # s^-1
OMEGA_M0 = 0.3
OMEGA_L = 0.7
SIGMA8_0 = 0.811


def H_of_z(z):
    return H0_SI * np.sqrt(OMEGA_M0*(1+z)**3 + OMEGA_L)


def Omega_m_of_z(z):
    return OMEGA_M0*(1+z)**3 / (OMEGA_M0*(1+z)**3 + OMEGA_L)


def g_cosmo(z, k_hMpc=0.1):
    """Cosmological 'gravitational acceleration' at scale k, redshift z."""
    h = H0_KMS_MPC/100.0
    k_phys = k_hMpc * h / 3.0857e22 * (1+z)
    return 1.5 * Omega_m_of_z(z) * H_of_z(z)**2 / k_phys


def mu_eff(z, beta, k_hMpc=0.1, alpha=ALPHA_SPHERICAL):
    """Effective gravitational coupling for linear perturbations."""
    x = g_cosmo(z, k_hMpc) / A0
    return 1.0 + (alpha*beta/2.0) / np.sqrt(x*(1.0+x))


def growth_solver(beta, k_hMpc=0.1, z_ini=50, alpha=ALPHA_SPHERICAL):
    """Solve linear growth equation in DEV.

        delta'' + (2 + H'/H) delta' - (3/2) Om(a) mu_eff delta = 0
    (primes are derivatives w.r.t. ln a)
    Returns scipy OdeResult with sol(lna).
    """
    def rhs(lna, y):
        a = np.exp(lna)
        z = 1.0/a - 1.0
        H = H_of_z(z)
        dH_dlna = -1.5*OMEGA_M0/a**3 * H0_SI**2 / H
        Hpr = dH_dlna / H
        d, dd = y
        mu = mu_eff(z, beta, k_hMpc, alpha)
        return [dd, -(2 + Hpr)*dd + 1.5*Omega_m_of_z(z)*mu*d]

    a_i = 1.0/(1+z_ini)
    return solve_ivp(rhs, [np.log(a_i), 0.0], [a_i, a_i],
                     dense_output=True, rtol=1e-10, atol=1e-13,
                     max_step=0.05)


def f_sigma8(beta, z_obs, sigma8_0=SIGMA8_0, k_hMpc=0.1,
             z_ini=50, alpha=ALPHA_SPHERICAL):
    """Compute f*sigma_8(z) for given beta at requested redshifts."""
    sol = growth_solver(beta, k_hMpc, z_ini, alpha)
    z_obs = np.atleast_1d(z_obs)
    lna_obs = np.log(1.0/(1+z_obs))
    y = sol.sol(lna_obs)
    delta, ddelta = y[0], y[1]
    f = ddelta / delta
    delta_today = sol.sol(0.0)[0]
    s8_z = sigma8_0 * delta/delta_today
    return f * s8_z


# Default fsigma8 dataset (DESI + BOSS + eBOSS + 6dFGS+MGS)
FSIGMA8_SURVEYS = [
    ("6dFGS+MGS", 0.15, 0.490, 0.045),
    ("BOSS LOWZ", 0.38, 0.497, 0.045),
    ("BOSS CMASS",0.51, 0.458, 0.038),
    ("eBOSS LRG", 0.70, 0.473, 0.044),
    ("eBOSS QSO", 1.48, 0.462, 0.045),
    ("DESI BGS",  0.51, 0.484, 0.044),
    ("DESI LRG",  0.93, 0.434, 0.041),
]


def chi2_fsigma8(beta, surveys=FSIGMA8_SURVEYS, **kw):
    z = np.array([s[1] for s in surveys])
    o = np.array([s[2] for s in surveys])
    sig = np.array([s[3] for s in surveys])
    pred = f_sigma8(beta, z, **kw)
    return float(np.sum(((o - pred)/sig)**2))


if __name__ == "__main__":
    print("DEV cosmology — fsigma8 baseline")
    for name, z, o, s in FSIGMA8_SURVEYS:
        pred = float(f_sigma8(0.0075, z))
        print(f"  {name:<12} z={z:.2f}  obs={o:.3f}+/-{s:.3f}  DEV={pred:.3f}")
    print(f"\nchi2_DEV  = {chi2_fsigma8(0.0075):.3f}")
    print(f"chi2_LCDM = {chi2_fsigma8(0.0):.3f}")
