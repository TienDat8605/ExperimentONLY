# Workspace contract

One job: maintain the evidence chain from ADBD implementation to experiments to manuscript.

## Ownership

| Material | Source of truth |
|---|---|
| Method behavior | `ONLY/patches/`, `ONLY/only_utils/`, and LLaVA integration |
| Run configuration | Exact runner command and evaluator arguments |
| Measurements | Raw artifacts under `logs/`, with provenance and coverage checked |
| Current paper | `paper/ACOMPA26_P02/main.tex` and the sections it actually includes |
| Earlier papers | `paper/versions/`, preserved independently |
| Navigation | `AGENTS.md`; contracts route to content rather than duplicating it |

`/home/dat/HCMUS/experiment` is separately tracked and is not a runtime dependency.
Its Proposal 4 uses layer-local consensus; this repository's Proposal 6 is ADBD.
Do not transfer proposal labels, thresholds, or results between them without checking code and metrics.

## 01 — Method

Inputs: `docs/method/CONTEXT.md`, `ONLY/CONTEXT.md`, and responsible source modules.
Process: inspect actual behavior, edit the narrow implementation surface, run relevant existing checks.
Outputs: source changes and verification evidence; use `tests/CONTEXT.md` for focused commands.
Human check: review the implementation and checks before using a changed method in an evaluation.

## 02 — Evaluation

Inputs: selected root runner, `ONLY/eval_bench/CONTEXT.md`, `data/CONTEXT.md`, code revision and seed.
Process: run only the authorized benchmark/configuration; preserve complete raw outputs.
Outputs: existing evaluator paths in `logs/`; create `logs/records/<run-id>.md` by copying
`_templates/experiment-record.md` when starting a recorded run. Create that folder on first use.
Human check: verify split coverage, completion, metric definition, configuration, and provenance
before approving a run for manuscript use. Recovered and partial logs retain those labels.

## 03 — Manuscript

Inputs: `paper/ACOMPA26_P02/CONTEXT.md`, included section(s), and reviewed experiment records/raw evidence.
Process: edit the selected sections, trace every changed number to evidence, build from the paper folder.
Outputs: updated LaTeX and compiled paper; build intermediates belong in a build directory.
Human check: read the changed text, tables, and rendered output before submission or publication.

## State and boundaries

Derive available state from files: code/tests, raw logs, reviewed records, LaTeX, and compiled outputs.
An existing file or PDF is not proof that an experiment is complete or a claim is verified.
Load only task-relevant inputs. Keep downloaded weights/assets apart from code and paper products.
Root runtime entry points and `ONLY/`, `data/`, `models/`, `logs/` paths remain stable for scripts.
Existing manuscript issues are recorded in `docs/reorganization/paper-readout.md`.
