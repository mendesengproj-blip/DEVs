# DEV Paper — Instruções de Compilação

## Arquivos

- `dev_paper.tex`  — corpo principal do paper
- `dev_refs.bib`   — referências BibTeX
- `figures/`       — figuras geradas pelo pipeline Python

## Como compilar

### Opção A — Overleaf (recomendado, sem instalação)

1. Acesse https://overleaf.com e crie um projeto novo
2. Faça upload de `dev_paper.tex` e `dev_refs.bib`
3. Crie uma pasta `figures/` e faça upload das figuras do pipeline:
   - fig2_rar.png
   - fig3_slip_signature.png
   - fig5_fisher_forecast.png
4. No menu Overleaf, selecione compilador: **pdfLaTeX**
5. Clique em Recompile

### Opção B — LaTeX local (Windows)

Instale MiKTeX: https://miktex.org/download

No terminal, dentro da pasta do paper:
```
pdflatex dev_paper.tex
bibtex dev_paper
pdflatex dev_paper.tex
pdflatex dev_paper.tex
```

## Figuras necessárias

Copie da pasta `dev_pipeline/figures/`:
- fig2_rar.png          → RAR com dados SPARC
- fig3_slip_signature.png → assinatura DEV em UDGs
- fig5_fisher_forecast.png → forecast de detectabilidade

## Próximos passos antes de submeter

1. [ ] Adicionar nome e afiliação no campo [Author]/[Institution]
2. [ ] Revisar todos os valores numéricos contra os CSVs gerados
3. [ ] Adicionar figura de curvas de rotação individuais (fig4)
4. [ ] Verificar referências — confirmar DOIs corretos
5. [ ] Rodar checagem de consistência: python run_analysis.py
6. [ ] Submeter ao arXiv (hep-ph ou astro-ph.GA) para estabelecer
       prioridade, depois ao journal (PRD ou JCAP recomendados)

## Revista alvo

**Primeira opção:** JCAP (Journal of Cosmology and Astroparticle Physics)
- Escopo exato para teoria modificada de gravidade + astrofísica
- Tempo de revisão típico: 6-10 semanas

**Segunda opção:** Physical Review D
- Maior impacto, revisão mais rigorosa
- Requer seção de comparação com outros modelos SVT mais detalhada

## Nota sobre o arXiv

Antes de submeter ao journal, poste no arXiv:
- Categoria: astro-ph.GA (primária) + gr-qc (secundária)
- Isso estabelece prioridade da data e permite feedback da comunidade
