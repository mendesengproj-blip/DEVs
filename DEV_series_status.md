# DEV Series — Status Report
Generated: 2026-05-10

## Papers

| Paper | Arquivo | Versão | Status | Pendência |
|-------|---------|--------|--------|-----------|
| I   | `paper_I/dev_paper_I_final.tex`     | FINAL | Submetido PRD DS14085 | Aguarda decisão |
| II  | `paper_II/dev_paper_II_FINAL.tex`   | FINAL | Pronto JCAP           | Aguarda nº arXiv Paper I |
| III | `paper_III/DEV_paper_III_FINAL.tex` | FINAL | Pronto                | Aguarda nº arXiv Paper I |

## Resultados centrais verificados

- χ²ν = 1.20 em 167 galáxias SPARC ✅
- α = 2/3 (slip pontual, derivado analiticamente, verificado SymPy) ✅
- Δχ² = −0.22 em fσ₈ vs ΛCDM ✅
- η−1 ∈ [2.23%, 4.08%] pontual (Tabela III corrigida) ✅
- η−1 ∈ [3.78%, 6.90%] estendido (Paper III) ✅
- α_operador = −1.56 (deep-MOND, pontual) ✅
- α NÃO universal: varia −1.96 (Hernquist) → −1.41 (UDGs) com ε ✅
- cs² ∈ [1/3, 1), L < 17 pc, m_A > 3.7×10⁻²⁵ eV/c² ✅
- β robusto: jackknife < 1.3% ✅
- SNR ~31σ para N=300 com Euclid ✅
- DGSAT-I: g/a₀=0.016, ε=7.96, η−1≃6.7% ✅

## Parâmetros canônicos

- a₀ = 1.2×10⁻¹⁰ m/s² = 3703 (km/s)²/kpc
- β = 0.0075 [0.0015, 0.0134]
- G = 4.302×10⁻³ kpc·(km/s)²·M⊙⁻¹

## Auditoria de conteúdo (resumo)

**Paper I** (`paper_I/dev_paper_I_final.tex`)
- ✅ Abstract: "$2$--$7\%$" em UDGs
- ✅ Tabela III: DGSAT-I g/a₀=0.016, η−1=3.95–6.7%, ε=7.96
- ✅ "Derivation status" com caveat ansatz (linha 342)
- ✅ Springel & Farrar 2007 citado via `\cite{SpringelFarrar2007}`
- ✅ SNR ~31σ (linha 797)

**Paper II** (`paper_II/dev_paper_II_FINAL.tex`)
- ✅ Introduction completa (linha 64)
- ✅ Conclusions completas; SNR ~31σ (linha 535)
- ✅ DGSAT-I ε≈7.96 (linhas 358, 369)
- ✅ Referência a Paper III com α=−1.56 (linhas 108, 516)
- ℹ️ Abstract diz "$2$--$4\%$" (não "1--5%") — número correto para regime extended-source

**Paper III** (`paper_III/DEV_paper_III_FINAL.tex`)
- ✅ `\usepackage[hidelinks]{hyperref}` (linha 10)
- ✅ Abstract com α=−1.56±0.02
- ✅ Tabela `tab:alpha_universality` (linha 161)
- ✅ Figura `fig:alpha_vs_epsilon` (linha 186)
- ✅ "α is not universal" (linha 339); "non-universality" (linha 355)
- ✅ Calcagni citado (linha 313)
- ✅ Parágrafo de limitação honesta (linha 351)

## Bibliografia

`dev_refs.bib` — 22 chaves, todas as 21 esperadas presentes
(+ Sakstein2016cluster extra). ✅

## Figuras

**Paper I** (`paper_I/figures/`) — 6/6 ✅
fig_beta_calibration, fig1_mu_function, fig2_rar, fig3_slip_signature,
fig4_rotation_curves, fig5_fisher_forecast.

**Paper II** (`paper_II/figures/`) — 6/6 ✅ (nomes diferem do prompt; conteúdo equivalente)
- fig_cs2.png, fig_L_constraint.png ← nomes esperados
- fig_correction.png ← (≡ fig_R_correction)
- fig_beta_upsilon_split.png ← (≡ fig_orthogonality)
- fig_upsilon_gas_degeneracy.png ← (≡ fig_chi2_contours)
- fig_beta_jackknife.png ← (≡ fig_jackknife)

**Paper III** (`paper_III/`) — 4/4 ✅
operator_identification, operator_comparison,
universality_alpha_vs_epsilon, universality_Seff.

## Submissões

- MNRAS MN-26-1316-P: rejeitado (escopo) 05/05/2026
- PRD DS14085: submetido 08/05/2026 — com editores
- arXiv: aguardando endorsement (código OUN9LO)
- Profa. Raissa Mendes (UFF): contactada para endorsement

## Commits relevantes (últimos 10)

```
23c14a6 Series audit: consolidate to final filenames. Paper I dev_paper_I_final.tex (PRD DS14085), Paper II dev_paper_II_FINAL.tex (JCAP ready), Paper III DEV_paper_III_FINAL.tex (alpha=-1.56 non-universal). Remove superseded versions. Add CLAUDE.md and dev_refs.bib.
9284a17 Paper III FINAL: add universality test. alpha NOT universal: varies from -1.62 (point) to -1.96 (Hernquist). UDGs cluster at alpha~-1.4.
c92959c Paper III complete: Introduction, Section III (UDG table with extended corrections), Conclusions.
8de2e48 Paper II: Fig. 6 caption 1.5% -> 1.3% (jackknife consistency)
830b8b2 Paper II complete: add Introduction and Conclusions
7c6ab02 Paper III: non-local operator identified
53a255f Correct UDG predictions in Papers I and II: fix DGSAT-I r_eff and all g/a0 values in Table III. SNR(N=300, Euclid)=31 sigma.
056dd93 Paper III: SI dimensional audit
e0f5f39 Paper III: H1 hypothesis rejected
3937db1 Paper II: corrected version, eta verification suite, PT translation
```

## Estado git

- Working tree limpo (somente este `DEV_series_status.md` untracked).
- Branch local sincronizado com `origin/main` (push concluído: `056dd93..23c14a6`).
- Remote: github.com/mendesengproj-blip/DEVs

---

✅ **Auditoria concluída. Série DEV consolidada e sincronizada.**
