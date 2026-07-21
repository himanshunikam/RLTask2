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

- Font is Helvetica (Arial-like sans-serif) via `helvet` + `\sfdefault`.
- Results come from the seeds-0-9 evaluation (`eval_*_results.json`). Three tables:
  Table 2 = per-method averages over each method's solved seeds; Table 3 = fair
  comparison on the seeds all three solved (2,5,7,9); Table 4 = per-seed detail.
- To add plots (learning curve / trajectory), drop the image into `report/` and
  uncomment the `figure` block near the end of the Ergebnisse section.
