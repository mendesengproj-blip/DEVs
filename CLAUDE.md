# CLAUDE.md — DEV Series Context

## Estado dos papers (maio 2026)

- **Paper I** — `paper_I/dev_paper_I_final.tex`
  Submetido ao PRD, manuscript ID **DS14085** (08/05/2026).
  Tabela III corrigida: ε ∈ [4.56, 8.21]; DGSAT-I com g/a₀=0.016, η−1=3.95–6.7%.
  SNR ~31σ para N=300 com Euclid. Slip predito 2–7% em UDGs.

- **Paper II** — `paper_II/dev_paper_II_FINAL.tex`
  Pronto para submissão ao JCAP. Aguarda nº arXiv do Paper I para
  atualizar referência cruzada.
  Resultados: cs² ∈ [1/3,1), L<17 pc, m_A > 3.7×10⁻²⁵ eV/c²;
  β e Υ★ ortogonais; jackknife < 1.3%.

- **Paper III** — `paper_III/DEV_paper_III_FINAL.tex`
  Pronto. Operador não-local identificado: α = −1.56 ± 0.02
  (deep-MOND, fonte pontual). Teste de universalidade adicionado:
  α NÃO universal (varia de −1.96 Hernquist a −1.41 UDGs).
  Aguarda nº arXiv do Paper I.

## Parâmetros canônicos

- a₀ = 1.2×10⁻¹⁰ m/s² = **3703 (km/s)²/kpc**
- β = 0.0075 [0.0015, 0.0134]
- G = 4.302×10⁻³ kpc·(km/s)²·M⊙⁻¹
- α (analítico, slip pontual) = 2/3 (verificado SymPy)
- α (operador, deep-MOND) = −1.56

## Resultados centrais

- χ²ν = 1.20 em 167 galáxias SPARC (sem parâmetros livres globais)
- Δχ² = −0.22 em fσ₈ vs ΛCDM
- η−1 ∈ [2.23%, 4.08%] pontual; [3.78%, 6.90%] estendido
- DGSAT-I: ε=7.96, η−1≃6.7% (mais forte teste UDG)

## Submissões

- MNRAS MN-26-1316-P: rejeitado por escopo (05/05/2026)
- PRD DS14085: submetido 08/05/2026 — com editores
- arXiv: aguardando endorsement (código OUN9LO);
  Profa. Raissa Mendes (UFF) contactada

## Convenções do repositório

- Versões `_v2`, `_v3`, `_v4`, `_corrected`, `_complete` foram consolidadas
  para os nomes finais acima. Não recriar versões intermediárias.
- Bibliografia centralizada em `dev_refs.bib` (21 chaves esperadas).
- Figuras Paper I em `paper_I/figures/`; Paper II em `paper_II/figures/`;
  Paper III em `paper_III/` (root da pasta).
