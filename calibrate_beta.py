import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from theory import eta_dev, A0

ETA_CONSTRAINTS = [
    ('SDSS_galaxies',   1.0,        1.005,    0.040,    'Reyes+2010'),
    ('CFHTLenS',        0.5,        0.99,     0.05,     'Simpson+2013'),
    ('MACS_J1206',      0.01,       1.04,     0.07,     'Pizzuti+2016'),
    ('A1689_lens_dyn',  0.005,      1.08,     0.10,     'Sakstein+2016'),
    ('Coma_outer',      0.003,      1.10,     0.12,     'estimativa'),
]


def chi2_beta(beta, alpha=2.0/3.0):
    chi2 = 0.0
    for name, x, eta_obs, eta_err, ref in ETA_CONSTRAINTS:
        g = x * A0
        eta_pred = eta_dev(g, beta=beta, alpha=alpha)
        chi2 += ((eta_pred - eta_obs) / eta_err)**2
    return chi2


def fit_beta():
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(chi2_beta, bounds=(1e-5, 1.0), method='bounded')
    beta_best = res.x
    chi2_min = res.fun

    from scipy.optimize import brentq
    try:
        beta_hi = brentq(lambda b: chi2_beta(b) - chi2_min - 1,
                          beta_best, beta_best * 100)
        beta_lo = brentq(lambda b: chi2_beta(b) - chi2_min - 1,
                          beta_best * 0.01, beta_best)
    except Exception:
        beta_hi = beta_best * 1.5
        beta_lo = beta_best * 0.5

    return {
        'beta_best': beta_best,
        'beta_lo': beta_lo,
        'beta_hi': beta_hi,
        'chi2_min': chi2_min,
        'ndof': len(ETA_CONSTRAINTS) - 1,
        'chi2_red': chi2_min / max(1, len(ETA_CONSTRAINTS) - 1),
    }


if __name__ == "__main__":
    print("Calibrando beta com vinculos da literatura...\n")
    print(f"{'Sistema':<20} {'g/a0':<10} {'eta_obs':<12} {'eta_err':<10}")
    print("-" * 55)
    for name, x, e, err, ref in ETA_CONSTRAINTS:
        print(f"{name:<20} {x:<10.4f} {e:<12.3f} {err:<10.3f}")

    result = fit_beta()
    print(f"\n{'='*55}")
    print(f"Resultado da calibracao:")
    print(f"  beta = {result['beta_best']:.4f}  "
          f"[{result['beta_lo']:.4f}, {result['beta_hi']:.4f}]  (1-sigma)")
    print(f"  chi2_min = {result['chi2_min']:.2f}  "
          f"(ndof = {result['ndof']})")
    print(f"  chi2_red = {result['chi2_red']:.2f}")
    print(f"{'='*55}")

    print("\nPredicoes DEV para UDGs com beta calibrado:")
    print(f"{'UDG':<15} {'g/a0':<10} {'eta-1 pred':<15} {'detectavel Euclid?':<18}")
    print("-" * 60)
    udgs = [
        ('NGC1052-DF2', 0.048),
        ('NGC1052-DF4', 0.068),
        ('DF44',        0.016),
        ('DGSAT-I',     0.0025),
        ('VCC1287',     0.017),
        ('DF17',        0.006),
    ]
    beta = result['beta_best']
    for name, x in udgs:
        g = x * A0
        eta_m1 = float(eta_dev(g, beta=beta)) - 1
        detect = "SIM" if eta_m1 > 0.01 else ("MARGINAL" if eta_m1 > 0.005 else "NAO")
        print(f"{name:<15} {x:<10.4f} {eta_m1:<15.4f} {detect:<18}")
