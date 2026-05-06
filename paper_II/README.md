# DEV Paper II — Stability, Scale Constraints, and Parameter Robustness

Pacote completo do paper de seguimento da DEV.

## Arquivos do Paper (LaTeX)

- `dev_paper_II.tex` — manuscrito completo em RevTeX 4-2 (~6 páginas)

**Para compilar no Overleaf:**
1. Cria projeto novo no Overleaf
2. Faz upload do `.tex` e dos 6 arquivos `.png` (todos juntos na raiz)
3. Compila — RevTeX já está configurado

## Figuras Geradas (PNG, 150 dpi)

| Arquivo | Descrição | Seção |
|---------|-----------|-------|
| `fig_cs2.png` | Sound speed $c_s^2(X)$ | II — Estabilidade |
| `fig_correction.png` | Fator de correção $R(r/L)$ | III — Escala $L$ |
| `fig_L_constraint.png` | Constraint em $L$ do SPARC | III — Escala $L$ |
| `fig_beta_upsilon_split.png` | Ortogonalidade $\beta$-$\Upsilon$ | IV — Degenerações |
| `fig_upsilon_gas_degeneracy.png` | Degenerescência $\Upsilon$-gás | IV — Degenerações |
| `fig_beta_jackknife.png` | Robustez de $\beta$ | IV — Degenerações |

## Pipeline de Análise (Python)

Os três módulos abaixo geram todos os resultados e figuras do paper.
Reproduzíveis com `numpy`, `scipy`, `sympy`, `matplotlib`.

| Módulo | Conteúdo |
|--------|----------|
| `stability.py` | $c_s^2$ analítico (SymPy), no-ghost, contagem DOF do Proca |
| `vector_scale.py` | Propagador exato do vetor, $R(r/L)$, constraint em $L$ |
| `degeneracies.py` | Fisher matrix, mapas $\chi^2$, jackknife do $\beta$ |

**Para reproduzir todas as figuras:**
```bash
python stability.py
python vector_scale.py
python degeneracies.py
```

## Resultados Principais

### Item 1 — Estabilidade
- $c_s^2(X) = (X^2 + X_0^2)/(X^2 + 3X_0^2) \in [1/3, 1)$
- No-ghost satisfeito identicamente
- Vetor Proca: 3 DOF, 0 ghosts

### Item 2 — Escala L
- $L < 17$ pc do SPARC (1% de precisão)
- $m_A > 3.7 \times 10^{-25}$ eV/$c^2$ (para $K=1$)
- Aproximação gradiente válida em todas as escalas relevantes

### Item 3 — Degenerações
- $\beta$ e $\Upsilon_*$ observacionalmente ortogonais
- Degenerescência $\Upsilon$-gás apenas em anãs ricas em gás (esperada)
- $\beta$ jackknife: variação máxima 1.3%

## Próximos Passos Sugeridos

1. Aguardar aceitação do paper original no arXiv
2. Compilar paper II no Overleaf, gerar PDF
3. Submeter ao arXiv assim que houver número do paper I para citação
4. Categoria sugerida: gr-qc (primary), astro-ph.GA (cross-list)
5. Mesmo código de endosso ULX944 vale para submissões adicionais

## Citação ao Paper I

A linha `\bibitem{Mendes2026}` do `.tex` precisa ser atualizada
com o número arXiv real do paper I assim que disponível.
Procura por `arXiv:XXXX.XXXXX` no `.tex` e substitui.

---

Miqueias Alves Mendes
mendes.eng.proj@gmail.com
https://github.com/mendesengproj-blip/DEVs
