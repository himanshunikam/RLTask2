# Task 2 Report (LaTeX, German)

`main.tex` is the full report. It uses only standard packages (helvet, babel/ngerman,
booktabs, geometry, hyperref) and an inline bibliography, so it compiles with plain
`pdflatex` — no bibtex/biber step needed.

## Compile

```bash
cd report
pdflatex main.tex
pdflatex main.tex     # run twice so references/section numbers resolve
```

This produces `main.pdf`. Alternatively: `latexmk -pdf main.tex`.

## Editing notes

- Styled to match the Task 1 report: serif font (`lmodern`), blue section headings
  (`secblue` via `titlesec`), grey title block with Matrikelnummer.
- Covers the three submitted methods: stochastic Actor-Critic, DDPG, TD3.
- Results come from the seeds-0-9 evaluation (`eval_*_results.json`). DDPG and TD3
  are filled in; the stochastic Actor-Critic rows say "offen" and need updating once
  its evaluation finishes (Table 1 hyperparameters, Table 2 results, and optionally a
  third column in Table 3 per-seed).
- Section 9 (Quellen) is intentionally empty and needs to be filled in.
- To add plots (learning curve / trajectory), drop the image into `report/` and
  uncomment the `figure` block near the end of the Ergebnisse section.
