"""
UDG Module
==========
Predições da DEV para galáxias ultra-difusas (UDGs) e calibração do
parâmetro de acoplamento beta.

Background físico:
- UDGs têm aceleração interna ~ 0.001-0.01 a0 (van Dokkum+ 2018)
- Regime de "vácuo profundamente saturado" — onde a DEV diverge mais
  do LambdaCDM e MOND padrão
- Sistemas conhecidos: NGC1052-DF2, DF4 (controversos), AGC 122966,
  e a amostra Coma (Yagi+2016)

A predição central:
    eta(g) - 1 ~ alpha * beta * sqrt(a0/g)  [regime profundo]

Esta é a única assinatura observacional que distingue DEV de:
- LambdaCDM (eta = 1)
- MOND/TeVeS (eta ~ 1, sem dependência de g)
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from theory import eta_dev, A0, G_NEWTON, KPC_TO_M, MSUN


@dataclass
class UDG:
    """Galáxia ultra-difusa com propriedades observacionais."""
    name: str
    M_star_msun: float          # massa estelar
    R_eff_kpc: float            # raio efetivo
    sigma_v_kms: float = None   # dispersão de velocidades (se medida)
    distance_mpc: float = 20.0
    g_internal: float = None    # aceleração interna característica


def compute_g_internal(udg: UDG):
    """
    Aceleração característica no raio efetivo de uma UDG.

    g ~ G * M_star / R_eff^2

    (Aproximação: assumimos M_dyn ~ M_star para UDGs típicas, antes
    da correção DEV. A própria correção é o que queremos prever.)
    """
    R_m = udg.R_eff_kpc * KPC_TO_M
    M_kg = udg.M_star_msun * MSUN
    return G_NEWTON * M_kg / R_m**2


# ============================================================
# Catálogo de UDGs reais conhecidas (parâmetros literatura)
# ============================================================
def real_udg_sample():
    """
    Amostra de UDGs com parâmetros da literatura.

    Referências:
    - DF2, DF44: van Dokkum et al. 2018, 2019
    - DGSAT I:   Martinez-Delgado+ 2016
    - VCC 1287:  Beasley+ 2016
    """
    return [
        UDG(name="NGC1052-DF2", M_star_msun=2.0e8, R_eff_kpc=2.2,
            sigma_v_kms=8.5, distance_mpc=20.0),
        UDG(name="NGC1052-DF4", M_star_msun=1.5e8, R_eff_kpc=1.6,
            sigma_v_kms=4.2, distance_mpc=20.0),
        UDG(name="DF44",        M_star_msun=3.0e8, R_eff_kpc=4.7,
            sigma_v_kms=33.0, distance_mpc=100.0),
        UDG(name="DGSAT-I",     M_star_msun=4.7e7, R_eff_kpc=4.7,
            distance_mpc=78.0),
        UDG(name="VCC1287",     M_star_msun=2.8e8, R_eff_kpc=4.4,
            sigma_v_kms=33.0, distance_mpc=16.5),
        UDG(name="DF17",        M_star_msun=1.0e8, R_eff_kpc=4.4,
            distance_mpc=100.0),
    ]


# ============================================================
# Predições DEV para slip
# ============================================================
def predict_eta_for_udgs(udgs: List[UDG], beta: float, alpha: float = 1.0):
    """
    Calcula eta(g) para cada UDG.

    Retorna lista de dicionários com predições.
    """
    results = []
    for udg in udgs:
        g_int = compute_g_internal(udg)
        eta_pred = eta_dev(g_int, beta, alpha)
        results.append({
            'name': udg.name,
            'M_star': udg.M_star_msun,
            'R_eff': udg.R_eff_kpc,
            'g_internal': g_int,
            'g_over_a0': g_int / A0,
            'eta_predicted': float(eta_pred),
            'slip_percent': float((eta_pred - 1.0) * 100),
        })
    return results


# ============================================================
# Calibração de beta usando observações de lensing/dinâmica
# ============================================================
def calibrate_beta_from_constraint(eta_observed, eta_err, g_observed,
                                    alpha=1.0):
    """
    Dado um valor observado de eta em uma única aceleração g,
    inverte para obter beta.

    eta - 1 = alpha * beta * f(g/a0)
    onde f(x) = 1/sqrt(x + x^2)

    Retorna beta_best e sigma_beta.
    """
    x = g_observed / A0
    f_x = 1.0 / np.sqrt(x + x**2)

    beta_best = (eta_observed - 1.0) / (alpha * f_x)
    beta_err = eta_err / (alpha * f_x)

    return beta_best, beta_err


def fisher_forecast(udgs: List[UDG], beta_true=0.01, alpha=1.0,
                    eta_err_per_udg=0.02):
    """
    Forecast de Fisher: dada uma amostra de UDGs com erro tipico
    eta_err em cada uma, qual a precisao em beta?

    sigma(beta)^-2 = sum_i [df_i/dbeta]^2 / sigma_i^2
                   = sum_i [alpha * f(x_i)]^2 / sigma_eta_i^2
    """
    fisher_info = 0.0
    for udg in udgs:
        g = compute_g_internal(udg)
        x = g / A0
        f_x = 1.0 / np.sqrt(x + x**2)
        fisher_info += (alpha * f_x)**2 / eta_err_per_udg**2

    sigma_beta = 1.0 / np.sqrt(fisher_info)
    snr = beta_true / sigma_beta

    return {
        'sigma_beta': sigma_beta,
        'snr': snr,
        'detection_at_5sigma': snr >= 5,
        'n_udgs': len(udgs),
    }


# ============================================================
# Comparação com modelos competidores
# ============================================================
def model_comparison_table(udgs: List[UDG], beta=0.01, alpha=1.0):
    """
    Tabela comparando predições de:
      - LambdaCDM:    eta = 1.0 (sem slip)
      - MOND padrão:  eta = 1.0 (sem slip)
      - DEV:          eta = 1 + alpha*beta*f(g/a0)
    """
    rows = []
    for udg in udgs:
        g = compute_g_internal(udg)
        eta_d = float(eta_dev(g, beta, alpha))
        rows.append({
            'name': udg.name,
            'g/a0': g/A0,
            'LambdaCDM': 1.0,
            'MOND': 1.0,
            'DEV': eta_d,
            'DEV-LambdaCDM': eta_d - 1.0,
        })
    return rows


if __name__ == "__main__":
    udgs = real_udg_sample()

    print("=" * 70)
    print("Predições DEV para UDGs reais (assumindo beta=0.01, alpha=1)")
    print("=" * 70)
    print(f"\n{'Galaxy':<14} {'M*[Msun]':<11} {'g/a0':<10} {'eta-1':<10} {'slip%':<8}")
    print("-" * 70)

    preds = predict_eta_for_udgs(udgs, beta=0.01)
    for p in preds:
        print(f"{p['name']:<14} {p['M_star']:.2e}   "
              f"{p['g_over_a0']:<10.4f} {p['eta_predicted']-1:<10.4f} "
              f"{p['slip_percent']:<8.2f}")

    print()
    print("=" * 70)
    print("Comparação entre modelos")
    print("=" * 70)
    print(f"\n{'Galaxy':<14} {'g/a0':<10} {'LCDM':<8} {'MOND':<8} {'DEV':<10} {'Δ(DEV-LCDM)':<12}")
    print("-" * 70)
    for row in model_comparison_table(udgs, beta=0.01):
        print(f"{row['name']:<14} {row['g/a0']:<10.4f} "
              f"{row['LambdaCDM']:<8.3f} {row['MOND']:<8.3f} "
              f"{row['DEV']:<10.4f} {row['DEV-LambdaCDM']:<12.4f}")

    print()
    print("=" * 70)
    print("Forecast de Fisher: detectabilidade do sinal DEV")
    print("=" * 70)
    for sigma_eta in [0.05, 0.02, 0.01]:
        forecast = fisher_forecast(udgs, beta_true=0.01,
                                    eta_err_per_udg=sigma_eta)
        print(f"\nCom erro sigma(eta)={sigma_eta} por UDG ({forecast['n_udgs']} UDGs):")
        print(f"  sigma(beta) = {forecast['sigma_beta']:.5f}")
        print(f"  SNR = {forecast['snr']:.2f}")
        print(f"  Detecção 5-sigma? {forecast['detection_at_5sigma']}")
