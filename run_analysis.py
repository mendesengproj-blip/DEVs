"""
Run Analysis: Pipeline DEV Completo
====================================
Script principal que:
  1. Constrói amostra SPARC sintética (substitua por dados reais)
  2. Ajusta cada galáxia individualmente
  3. Constrói a Radial Acceleration Relation (RAR)
  4. Faz predições para UDGs reais
  5. Forecast de Fisher para detectabilidade
  6. Gera figuras-chave do paper

Saída: figuras em PNG e tabela CSV de resultados.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from theory import (mu_dev, nu_dev, eta_dev, v_circ_dev,
                    A0, X0, KPC_TO_M, MSUN, G_NEWTON)
from sparc import (load_sparc_folder, fit_galaxy, fit_all,
                    compute_rar, Galaxy)
from udg import (real_udg_sample, predict_eta_for_udgs,
                  fisher_forecast, compute_g_internal,
                  model_comparison_table)

# Estética das figuras
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})

OUTDIR = "figures"
import os
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# Figura 1: Função de interpolação mu(x)
# ============================================================
def fig1_mu_function():
    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.logspace(-3, 3, 500)
    ax.loglog(x, mu_dev(x), 'k-', lw=2.2, label=r'DEV: $\mu(x)=x/\sqrt{1+x^2}$')

    ax.loglog(x[x < 0.1], x[x < 0.1], 'b--', lw=1, alpha=0.6,
              label=r'Deep MOND: $\mu \to x$')
    ax.loglog(x[x > 10], np.ones_like(x[x > 10]), 'r--', lw=1, alpha=0.6,
              label=r'Newton: $\mu \to 1$')

    ax.axvline(1, color='gray', ls=':', alpha=0.5)
    ax.text(1.05, 0.005, r'$g = a_0$', rotation=90, alpha=0.7)

    ax.set_xlabel(r'$x = g / a_0$')
    ax.set_ylabel(r'$\mu(x)$')
    ax.set_title('DEV interpolation function from DBI Lagrangian')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_xlim(1e-3, 1e3)
    ax.set_ylim(1e-3, 2)

    plt.savefig(f"{OUTDIR}/fig1_mu_function.png")
    plt.close()
    print(f"[OK] Figura 1 salva: {OUTDIR}/fig1_mu_function.png")


# ============================================================
# Figura 2: Radial Acceleration Relation (RAR)
# ============================================================
def fig2_rar(galaxies):
    g_bar, g_obs, g_err = compute_rar(galaxies)

    fig, ax = plt.subplots(figsize=(7.5, 6))

    # Pontos observacionais
    ax.errorbar(g_bar, g_obs, yerr=g_err, fmt='o', ms=4, alpha=0.5,
                color='steelblue', label=f'SPARC galaxies ($n={len(galaxies)}$)')

    g_grid = np.logspace(-13, -8, 200)
    g_pred = nu_dev(g_grid / A0) * g_grid
    ax.loglog(g_grid, g_pred, 'k-', lw=2,
              label='DEV prediction (zero global free parameters)')

    ax.loglog(g_grid, g_grid, 'r--', lw=1, alpha=0.6,
              label=r'Newton ($g_{\rm obs}=g_{\rm bar}$)')

    ax.axhline(A0, color='gray', ls=':', alpha=0.5)
    ax.axvline(A0, color='gray', ls=':', alpha=0.5)
    ax.text(A0*1.5, 1e-13, r'$a_0$', alpha=0.6)

    ax.set_xlabel(r'$g_{\rm bar}$ [m s$^{-2}$]')
    ax.set_ylabel(r'$g_{\rm obs}$ [m s$^{-2}$]')
    ax.set_title(f'Radial Acceleration Relation: DEV prediction vs SPARC data ($n={len(galaxies)}$)')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_xlim(1e-13, 1e-8)
    ax.set_ylim(1e-13, 1e-8)
    ax.set_aspect('equal')

    plt.savefig(f"{OUTDIR}/fig2_rar.png")
    plt.close()
    print(f"[OK] Figura 2 salva: {OUTDIR}/fig2_rar.png")


# ============================================================
# Figura 3: Slip gravitacional eta(g) — A ASSINATURA DEV
# ============================================================
def fig3_slip_signature():
    fig, ax = plt.subplots(figsize=(8, 6))

    g_grid = np.logspace(-14, -8, 300)
    x_grid = g_grid / A0

    # Bandas de beta variando (incerteza de calibração)
    for beta, ls, label in [(0.005, '--', r'$\beta=0.005$'),
                              (0.01, '-', r'$\beta=0.01$'),
                              (0.02, '-.', r'$\beta=0.02$')]:
        eta = eta_dev(g_grid, beta=beta) - 1.0
        ax.loglog(x_grid, eta, ls, lw=2, label=f'DEV ({label})')

    ax.axhline(1e-4, color='red', alpha=0.0)
    ax.text(1e-5, 5e-4, r'$\Lambda$CDM/MOND: $\eta - 1 = 0$',
            color='gray', alpha=0.8, fontsize=10)

    # Marcar UDGs reais
    udgs = real_udg_sample()
    for udg in udgs:
        g = compute_g_internal(udg) / A0
        eta_at = eta_dev(compute_g_internal(udg), beta=0.01) - 1.0
        ax.scatter(g, eta_at, s=80, marker='*', zorder=5,
                   color='darkorange', edgecolors='black', lw=0.8)
        ax.annotate(udg.name, (g, eta_at),
                    xytext=(8, 5), textcoords='offset points',
                    fontsize=8.5, alpha=0.9)

    # Sensibilidade observacional típica (Euclid ~ 1-5%)
    ax.axhspan(0.01, 0.05, alpha=0.12, color='green',
                label='Euclid sensitivity')

    ax.set_xlabel(r'$g / a_0$')
    ax.set_ylabel(r'$\eta - 1 \equiv \Phi/\Psi - 1$')
    ax.set_title('Gravitational slip: unique signature of DEV')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=9.5)
    ax.set_xlim(1e-5, 1e3)
    ax.set_ylim(1e-5, 5)

    plt.savefig(f"{OUTDIR}/fig3_slip_signature.png")
    plt.close()
    print(f"[OK] Figura 3 salva: {OUTDIR}/fig3_slip_signature.png")


# ============================================================
# Figura 4: Curvas de rotação para 4 galáxias representativas
# ============================================================
def fig4_rotation_curves(galaxies):
    # Selecionar 4 galáxias cobrindo toda a faixa de massa
    masses = [g.M_bar_enclosed[-1] for g in galaxies]
    order = np.argsort(masses)
    selected = [galaxies[order[0]], galaxies[order[len(order)//3]],
                galaxies[order[2*len(order)//3]], galaxies[order[-1]]]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, gal in zip(axes.flat, selected):
        # Observação
        ax.errorbar(gal.r_kpc, gal.v_obs_kms, yerr=gal.v_err_kms,
                     fmt='o', ms=5, color='steelblue', label='SPARC observation')

        v_pred = v_circ_dev(gal.r_kpc, gal.M_bar_enclosed)
        ax.plot(gal.r_kpc, v_pred, 'k-', lw=2, label='DEV')

        r_m = gal.r_kpc * KPC_TO_M
        v_newton = np.sqrt(G_NEWTON * gal.M_bar_enclosed * MSUN / r_m) / 1000
        ax.plot(gal.r_kpc, v_newton, 'r--', lw=1.2, alpha=0.7,
                label='Newton (baryons only)')

        ax.set_xlabel(r'$R$ [kpc]')
        ax.set_ylabel(r'$v_{\rm circ}$ [km s$^{-1}$]')
        ax.set_title(f'{gal.name}  ($M_{{\\rm bar}} = {gal.M_bar_enclosed[-1]:.1e}\\,M_\\odot$)')
        ax.legend(fontsize=9)

    fig.suptitle('Representative rotation curves: DEV fit to SPARC data', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig4_rotation_curves.png")
    plt.close()
    print(f"[OK] Figura 4 salva: {OUTDIR}/fig4_rotation_curves.png")


# ============================================================
# Figura 5: Forecast de Fisher
# ============================================================
def fig5_fisher_forecast():
    udgs = real_udg_sample()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Painel esquerdo: SNR vs número de UDGs (com erro fixo)
    ax = axes[0]
    n_udgs_grid = np.arange(1, 21)
    for sigma_eta, color, label in [(0.05, 'orange', r'$\sigma_\eta=0.05$ (current)'),
                                       (0.02, 'red', r'$\sigma_\eta=0.02$ (Euclid)'),
                                       (0.01, 'purple', r'$\sigma_\eta=0.01$ (future)')]:
        snrs = []
        for n in n_udgs_grid:
            sample = (udgs * (n // len(udgs) + 1))[:n]
            f = fisher_forecast(sample, beta_true=0.01, eta_err_per_udg=sigma_eta)
            snrs.append(f['snr'])
        ax.plot(n_udgs_grid, snrs, 'o-', color=color, label=label, lw=2, ms=4)

    ax.axhline(5, color='black', ls='--', alpha=0.5, label=r'5$\sigma$ detection')
    ax.axhline(3, color='black', ls=':', alpha=0.5, label=r'3$\sigma$ detection')
    ax.set_xlabel('Number of UDGs with weak lensing')
    ax.set_ylabel(r'SNR for $\beta = 0.01$')
    ax.set_title('Fisher forecast: DEV detectability')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_yscale('log')

    # Painel direito: predição eta para cada UDG real
    ax = axes[1]
    preds = predict_eta_for_udgs(udgs, beta=0.01)
    names = [p['name'] for p in preds]
    slips = [p['slip_percent'] for p in preds]
    g_a0 = [p['g_over_a0'] for p in preds]

    bars = ax.barh(names, slips, color='steelblue', edgecolor='black')
    ax.axvline(0, color='gray', alpha=0.5)
    ax.axvline(1, color='green', ls='--', alpha=0.6, label=r'$\sim$1% sensitivity')
    ax.axvline(5, color='red', ls='--', alpha=0.6, label=r'$\sim$5% sensitivity')
    ax.set_xlabel(r'Predicted gravitational slip $(\eta-1)\times 100\%$')
    ax.set_title(r'DEV slip predictions for known UDGs ($\beta=0.01$)')
    ax.legend(fontsize=9)

    # Anotar g/a0 ao lado de cada barra
    for bar, val, x in zip(bars, slips, g_a0):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f'g/a₀ = {x:.3f}', va='center', fontsize=8.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig5_fisher_forecast.png")
    plt.close()
    print(f"[OK] Figura 5 salva: {OUTDIR}/fig5_fisher_forecast.png")


# ============================================================
# Tabela resumo
# ============================================================
def generate_results_table(galaxies, udgs):
    # Tabela 1: ajustes SPARC
    df_sparc = fit_all(galaxies, verbose=False)
    df_sparc.to_csv(f"{OUTDIR}/results_sparc.csv", index=False)
    print(f"[OK] Tabela SPARC salva: {OUTDIR}/results_sparc.csv")

    # Tabela 2: predições UDG
    preds = predict_eta_for_udgs(udgs, beta=0.01)
    df_udg = pd.DataFrame(preds)
    df_udg.to_csv(f"{OUTDIR}/results_udg.csv", index=False)
    print(f"[OK] Tabela UDG salva: {OUTDIR}/results_udg.csv")

    return df_sparc, df_udg


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("PIPELINE DEV: Análise Completa SPARC + UDG")
    print("=" * 70)

    # Construir amostras
    print("\n[1/6] Carregando catálogo SPARC real...")
    galaxies = load_sparc_folder("sparc_data")
    print(f"      n_galáxias = {len(galaxies)}")

    udgs = real_udg_sample()
    print(f"      n_UDGs = {len(udgs)}")

    # Figuras
    print("\n[2/6] Figura 1: função de interpolação mu(x)...")
    fig1_mu_function()

    print("\n[3/6] Figura 2: Radial Acceleration Relation...")
    fig2_rar(galaxies)

    print("\n[4/6] Figura 3: assinatura de slip eta(g)...")
    fig3_slip_signature()

    print("\n[5/6] Figura 4: curvas de rotação representativas...")
    fig4_rotation_curves(galaxies)

    print("\n[6/6] Figura 5: forecast de Fisher...")
    fig5_fisher_forecast()

    # Tabelas
    print("\n[Tabelas] Gerando CSVs de resultados...")
    df_sparc, df_udg = generate_results_table(galaxies, udgs)

    # Resumo final
    print("\n" + "=" * 70)
    print("RESUMO QUANTITATIVO DO PIPELINE")
    print("=" * 70)

    chi2_reds = df_sparc['chi2_red'].values
    print(f"\nSPARC:")
    print(f"  chi2_red mediano  = {np.median(chi2_reds):.3f}")
    print(f"  chi2_red médio    = {np.mean(chi2_reds):.3f}")
    print(f"  Frac com chi2_red < 1.5 = {np.mean(np.array(chi2_reds) < 1.5)*100:.0f}%")

    print(f"\nUDG (predições com beta=0.01):")
    preds = predict_eta_for_udgs(udgs, beta=0.01)
    slips = [p['slip_percent'] for p in preds]
    print(f"  Slip predito range: [{min(slips):.1f}%, {max(slips):.1f}%]")
    print(f"  Slip mediano:       {np.median(slips):.1f}%")

    forecast = fisher_forecast(udgs, beta_true=0.01, eta_err_per_udg=0.05)
    print(f"\nDetectabilidade ({forecast['n_udgs']} UDGs, sigma_eta=0.05):")
    print(f"  SNR = {forecast['snr']:.2f}  ({'DETECTÁVEL' if forecast['detection_at_5sigma'] else 'MARGINAL'})")

    print("\n" + "=" * 70)
    print(f"Todas as saídas em: {OUTDIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
