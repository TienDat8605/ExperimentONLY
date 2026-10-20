# Reorganization validation

The reorganization preserves 48 original files, checked against the pre-migration SHA-256 manifest.
`migration-result.json` records the verified local preservation bundle and unchanged user-script hashes.
The sibling repository was inventoried but not modified.

## Checks

- Manifest parity, compatibility symlink, local Markdown navigation, and active/historical LaTeX
  dependency graphs: `python3 docs/reorganization/verify.py` — passed; 48 hashes, four LaTeX
  entry points, and 20 Markdown files checked.
- CLI regression checks: `python3 -m unittest discover -s tests -p 'test_proposal4_local_cli.py'`
  — all 6 passed.
- Proposal 6 mask checks: `python3 -m unittest discover -s tests -p 'test_proposal6_mask.py'`
  — all 8 passed.
- Active manuscript: `latexmk -pdf -interaction=nonstopmode -halt-on-error
  -outdir=/tmp/experimentonly-acompa26-build main.tex` from `paper/ACOMPA26_P02/`.
  Build passed and produced a seven-page PDF. It retains the pre-existing missing
  `tab:detailed_results` reference, missing-author warnings, and layout warnings.

The source text and reported numbers were preserved. No GPU evaluation was launched.
See `paper-readout.md` for unresolved manuscript claims and evidence discrepancies.

## Cold walk

`AGENTS.md` routes directly to the current paper's contract, implementation contract,
benchmark contract, and evidence contract. Each names its owning sources and outputs.
An editor can find Proposal 6 behavior, its evaluator, the active LaTeX input graph,
and run provenance in the root router plus one contract.

## Git scope

Commit the organizational documentation, manuscript/reference relocations, compatibility symlink,
and ignore rules. Existing modifications to `colab.sh`, `recover_pope_log.py`, and
`run_proposal4_colab.sh`, plus the unrelated untracked recovered logs, remain outside this commit.
Local migration backups and new build products are ignored.
The historical `main-old.log` is explicitly retained with its snapshot so every manifest
destination remains available after cloning. Whitespace in original manuscripts/drafts is
preserved for byte parity; source-preservation warnings are not normalized away.
