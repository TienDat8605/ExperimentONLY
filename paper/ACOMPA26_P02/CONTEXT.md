# Active ACOMPA26_P02 paper

One job: write and build **Adaptive Dual-Branch Decoding for Hallucination Mitigation
in Large Vision–Language Models**.

## Inputs

`main.tex` is authoritative. Its ordered inputs are `abstract_.tex`, `introduction_.tex`,
`relatedwork_.tex`, `method_.tex`, `experiment_.tex`, `conclusion_.tex`, and `bibliography_.tex`.
The bibliography uses inline `thebibliography`; `relatedwork_.tex` reads `visual.png`.
Unsuffixed sections and `introduction(2)_.tex` are retained alternatives, not active inputs.
Load only the relevant included sections, responsible code, and reviewed raw evidence.

## Process and outputs

Edit included files without silently changing `main.tex` to select an alternative draft.
Trace implementation claims to `../../ONLY/CONTEXT.md` and numbers to `../../logs/CONTEXT.md`.
Build from this directory so relative inputs and the figure resolve:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Outputs: edited source and `build/main.pdf`; intermediates remain in `build/`.
Existing alternate drafts and earlier manuscripts are preserved for comparison.

## Human check

Read changed text/tables and the rendered PDF. Resolve the issues in
[../../docs/reorganization/paper-readout.md](../../docs/reorganization/paper-readout.md)
against evidence before publication: the MME total discrepancy, missing detailed-results label,
TBD claims, and unfinished analyses are pre-existing manuscript work, not migration changes.
