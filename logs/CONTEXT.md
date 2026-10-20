# Experiment evidence contract

One job: retain raw outputs and provenance for completed, partial, and recovered runs.

## Inputs and process

Inputs: the exact command, code revision (including uncommitted changes), seed, model/assets, and
benchmark configuration. Root runners and evaluators write their existing timestamped output paths.
Recovery utilities may produce separate recovered outputs; preserve those origins and coverage.
Do not rename historical output folders or promote a recovered fragment to a complete run.

## Outputs

- Existing raw logs, predictions, timestamped result directories, and recovery archives.
- For a new recorded run, copy [../_templates/experiment-record.md](../_templates/experiment-record.md)
  to `records/<run-id>.md`; create `records/` when first needed, and link the exact raw artifacts.
- Approved measurements are cited by the active paper; raw logs remain the owning evidence.

## Human check

Check completion, sample count, split, seed, coefficients, metric definitions, and source revision.
Approve evidence before importing it into tables. Paper numbers without a traceable run remain
unverified; folder names and proposal IDs alone are insufficient provenance.
