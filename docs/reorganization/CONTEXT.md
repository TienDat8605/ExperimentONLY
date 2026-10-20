# Organization audit and migration proof

One job: retain the reviewed plan, original inventory, and reproducible migration checks.

## Inputs and process

- `PLAN.md`: approved layout and migration sequence.
- `inventory.csv`, `references.csv`, `symlinks.csv`, `collisions.csv`, `audit-summary.json`:
  generated **pre-migration snapshots**, not live indexes; do not hand-edit them.
- `migration.csv`: original source → destination manifest with byte hashes.
- `migration-result.json`: actual completed migration and verified local backup identity.
- `paper-readout.md`: manuscript orientation and unresolved pre-existing issues.
- `backups/`: ignored local preservation bundle and original Git status, including untracked files.

Run `python3 docs/reorganization/verify.py` from the root to verify destination hashes, the
compatibility link, active and historical LaTeX inputs/figures, and local Markdown links.
`audit.py` generated the original bounded inventory and now routes to post-migration verification.
It does not merge the sibling repository or rewrite historical snapshot indexes.

## Outputs and human check

Outputs: verification report and `VALIDATION.md` for checks actually run. Review parity and the
Git diff before publishing changes. Manuscript warnings are reported separately from file migration.
