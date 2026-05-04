"""
SPARC Module (dados reais)
==========================
Lê o catálogo SPARC real (Lelli, McGaugh & Schombert 2016).

Formato esperado dos arquivos _rotmod.dat:
    Colunas: Rad  Vobs  errV  Vgas  Vdisk  Vbul  SBdisk  SBbul
    Unidades: kpc  km/s  km/s  km/s  km/s   km/s  L/pc²   L/pc²
"""

import numpy as np
import pandas as pd
import os
import glob
from dataclasses import dataclass
from typing import List, Optional, Tuple
from theory import v_circ_dev, A0, G_NEWTON, KPC_TO_M, MSUN


@dataclass
class Galaxy:
    name: str
    r_kpc:          np.ndarray
    v_obs_kms:      np.ndarray
    v_err_kms:      np.ndarray
    v_gas_kms:      np.ndarray
    v_disk_kms:     np.ndarray
    v_bul_kms:      np.ndarray
    M_bar_enclosed: np.ndarray
    distance_mpc:   float = 10.0
    galaxy_type:    str   = "spiral"
    ml_disk: float = 1.0
    ml_bul:  float = 0.7


def _parse_rotmod(filepath):
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    rows.append([float(p) for p in parts[:8]])
                except ValueError:
                    continue
    if len(rows) < 3:
        return None
    cols = ['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
    n = min(len(rows[0]), len(cols))
    df = pd.DataFrame(rows, columns=cols[:n])
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[(df['Vobs'] > 0) & (df['errV'] > 0) & (df['Rad'] > 0)]
    return df if len(df) >= 3 else None


def _compute_M_bar(r_kpc, v_gas, v_disk, v_bul, ml_disk, ml_bul):
    v_gas_ms  = np.abs(v_gas)  * 1000.0
    v_disk_ms = np.abs(v_disk) * 1000.0
    v_bul_ms  = np.abs(v_bul)  * 1000.0
    r_m = r_kpc * KPC_TO_M
    v_bar2 = v_gas_ms**2 + ml_disk * v_disk_ms**2 + ml_bul * v_bul_ms**2
    M_bar = v_bar2 * r_m / G_NEWTON
    return M_bar / MSUN


def load_sparc_folder(folder, min_points=5):
    pattern = os.path.join(folder, "*_rotmod.dat")
    files = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(folder, "*.dat")
        files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nenhum .dat em '{folder}'")

    galaxies, skipped = [], []
    for fpath in files:
        name = os.path.basename(fpath).replace('_rotmod.dat','').replace('.dat','')
        df = _parse_rotmod(fpath)
        if df is None or len(df) < min_points:
            skipped.append(name)
            continue
        v_max = float(df['Vobs'].max())
        if v_max < 30.0 and len(df) < 8:
            skipped.append(name)
            continue
        M_bar = _compute_M_bar(df['Rad'].values, df['Vgas'].values,
                                df['Vdisk'].values, df['Vbul'].values, 1.0, 0.7)
        galaxies.append(Galaxy(
            name=name,
            r_kpc=df['Rad'].values,
            v_obs_kms=df['Vobs'].values,
            v_err_kms=df['errV'].values,
            v_gas_kms=df['Vgas'].values,
            v_disk_kms=df['Vdisk'].values,
            v_bul_kms=df['Vbul'].values,
            M_bar_enclosed=M_bar,
        ))

    print(f"Carregadas: {len(galaxies)} | Ignoradas: {len(skipped)}")
    return galaxies


def _chi2_galaxy(galaxy, ml_disk, ml_bul):
    M_bar = _compute_M_bar(galaxy.r_kpc, galaxy.v_gas_kms,
                            galaxy.v_disk_kms, galaxy.v_bul_kms,
                            ml_disk, ml_bul)
    v_pred = v_circ_dev(galaxy.r_kpc, M_bar)
    v_err_eff = np.sqrt(galaxy.v_err_kms**2 + (0.05 * galaxy.v_obs_kms)**2)
    res = (galaxy.v_obs_kms - v_pred) / v_err_eff
    return float(np.sum(res**2))


def fit_galaxy(galaxy, n_grid=30):
    ml_d_grid = np.linspace(0.3, 5.0, n_grid)
    ml_b_grid = np.linspace(0.3, 3.0, n_grid)
    best_chi2, best_d, best_b = np.inf, 1.0, 0.7
    for md in ml_d_grid:
        for mb in ml_b_grid:
            c = _chi2_galaxy(galaxy, md, mb)
            if c < best_chi2:
                best_chi2, best_d, best_b = c, md, mb
    try:
        from scipy.optimize import minimize
        res = minimize(lambda p: _chi2_galaxy(galaxy, p[0], p[1]),
                       x0=[best_d, best_b],
                       bounds=[(0.3,5.0),(0.3,3.0)], method='L-BFGS-B')
        if res.success and res.fun < best_chi2:
            best_chi2, best_d, best_b = res.fun, *res.x
    except Exception:
        pass

    ndof = max(1, len(galaxy.r_kpc) - 2)
    galaxy.ml_disk = best_d
    galaxy.ml_bul  = best_b
    galaxy.M_bar_enclosed = _compute_M_bar(galaxy.r_kpc, galaxy.v_gas_kms,
                                            galaxy.v_disk_kms, galaxy.v_bul_kms,
                                            best_d, best_b)
    return {'name': galaxy.name, 'ml_disk': best_d, 'ml_bul': best_b,
            'chi2': best_chi2, 'chi2_red': best_chi2/ndof, 'ndof': ndof,
            'n_points': len(galaxy.r_kpc), 'v_max': galaxy.v_obs_kms.max()}


def fit_all(galaxies, verbose=True):
    results = []
    for i, gal in enumerate(galaxies):
        r = fit_galaxy(gal)
        results.append(r)
        if verbose and (i % 10 == 0 or i == len(galaxies)-1):
            print(f"  [{i+1:3d}/{len(galaxies)}] {gal.name:<22} "
                  f"chi2_red={r['chi2_red']:.3f}  ML_disk={r['ml_disk']:.2f}")
    return pd.DataFrame(results)


def compute_rar(galaxies):
    g_bar_all, g_obs_all, g_err_all = [], [], []
    for gal in galaxies:
        r_m     = gal.r_kpc * KPC_TO_M
        g_bar   = G_NEWTON * gal.M_bar_enclosed * MSUN / r_m**2
        v_ms    = gal.v_obs_kms  * 1000.0
        verr_ms = gal.v_err_kms  * 1000.0
        g_obs   = v_ms**2  / r_m
        g_err   = 2 * v_ms * verr_ms / r_m
        mask = (g_bar > 0) & (g_obs > 0) & np.isfinite(g_bar + g_obs)
        g_bar_all.extend(g_bar[mask])
        g_obs_all.extend(g_obs[mask])
        g_err_all.extend(g_err[mask])
    return np.array(g_bar_all), np.array(g_obs_all), np.array(g_err_all)


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "sparc_data"
    print(f"Carregando SPARC de: {folder}\n")
    galaxies = load_sparc_folder(folder)
    print(f"\nAjustando {len(galaxies)} galáxias...\n")
    df = fit_all(galaxies, verbose=True)
    print("\n" + "="*60)
    print("RESUMO")
    print("="*60)
    print(f"N galáxias       : {len(df)}")
    print(f"chi2_red mediano : {df['chi2_red'].median():.3f}")
    print(f"chi2_red médio   : {df['chi2_red'].mean():.3f}")
    print(f"chi2_red < 1.5   : {(df['chi2_red']<1.5).mean()*100:.0f}%")
    print(f"chi2_red < 2.0   : {(df['chi2_red']<2.0).mean()*100:.0f}%")
    print(f"\nPiores 5 ajustes:")
    print(df.nlargest(5,'chi2_red')[['name','chi2_red','ml_disk','n_points']].to_string(index=False))
    df.to_csv("results_sparc_v2.csv", index=False)
    print(f"\nSalvo em: results_sparc_v2.csv")