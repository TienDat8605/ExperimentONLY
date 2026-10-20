---
type: restructure-proposal
status: implemented
scope: ExperimentONLY
---

# Proposed ICM organization

The active paper is `ACOMPA26_P02/main.tex`, titled **Adaptive Dual-Branch
Decoding for Hallucination Mitigation in Large Vision–Language Models**.
Its actual inputs are the seven underscore-suffixed section files. Organize
the workspace around that paper, its implementation, and its evidence.

This reviewed proposal was implemented after the user's instruction to commit
and push the reorganization. See [migration-result.json](migration-result.json)
and [VALIDATION.md](VALIDATION.md) for execution and checks. The sections below
retain the approved migration design; CSV inventories describe the original state.

## Scope and form

Use a small workspace catalog and folder contracts, combining a code system
map with a manuscript version library. The existing source tree remains the
implementation authority; logs remain the evidence authority; the active
LaTeX input graph remains the manuscript authority. Number the documented
workflow steps (method → evaluation → manuscript); folders that hold different
kinds of material are named by their job rather than an artificial sequence.

`/home/dat/HCMUS/experiment` remains a separately tracked comparison repository.
It is inventoried and checked for textual references, but is not merged or
moved. Its Proposal 4 is layer-local consensus; this repository's Proposals
4/5/6 are complementary-branch variants. A later migration of the sibling
repository needs its own manifest and filesystem write permission.

## Target tree

```text
ExperimentONLY/
├── AGENTS.md                       small task router
├── CONTEXT.md                      ownership and three-step workflow
├── README.md                       user setup/run/build instructions
├── ONLY/                           existing implementation paths
│   ├── CONTEXT.md                  runtime patch and model boundaries
│   ├── patches/                    existing LLaMA patch
│   ├── only_utils/                 masks and decoding
│   ├── eval_bench/                 evaluation entry points
│   └── experiments/llava/          existing vendored model code
├── tests/                          existing verification paths + contract
├── data/                           existing downloaded asset paths + contract
├── models/                         existing weights path, if downloaded
├── logs/                           existing raw/recovered evidence + contract
├── paper/
│   ├── CONTEXT.md                  active version and evidence rules
│   ├── ACOMPA26_P02/               current root manuscript, byte-preserved
│   │   ├── CONTEXT.md              main.tex input graph and build directory
│   │   ├── main.tex
│   │   ├── abstract_.tex, ...      underscore variants (actual names unchanged)
│   │   ├── *.tex                   unsuffixed alternative drafts, retained
│   │   └── visual.png
│   └── versions/
│       ├── CONTEXT.md              earlier snapshots, not new editing targets
│       ├── initial/                previous paper/ACOMPA26_P02 contents
│       └── july-rewrite/           previous write-paper/ACOMPA26_P02 contents
├── docs/
│   ├── CONTEXT.md                  method reference and audit routing
│   ├── method/
│   │   ├── CONTEXT.md              distinguish references from historical drafts
│   │   ├── ONLY_report.md
│   │   ├── CMD_abstract_intro_draft.md
│   │   ├── CMD_method_draft.md
│   │   ├── CMD_experiments_draft.md
│   │   └── main_proposal6_rewrite_with_mme.md
│   └── reorganization/             this proposal and generated audit
├── _templates/
│   └── experiment-record.md        code revision, command, seed, outputs, review
├── ACOMPA26_P02 -> paper/ACOMPA26_P02  compatibility link for current paper path
└── existing root runners/utilities  retained, documented in root catalog
```

The underscore variants are the actual `abstract_.tex`, `method_.tex`, etc.;
no filename normalization is proposed. Preserve all
manuscript alternatives and build artifacts. Earlier snapshots are versioned
products, not files declared dead. No deletion or deduplication is proposed.

## Moves and reference handling

The complete file-level move list is [migration.csv](migration.csv), with
source SHA-256 and size. [references.csv](references.csv) enumerates candidate
referrers, including basename matches; it deliberately includes false positives
instead of treating a missing exact-path match as proof of safety.

| Order | Current path | Proposed path | Role and handling |
|---|---|---|---|
| 1 | `paper/ACOMPA26_P02/` | `paper/versions/initial/` | Earlier product; preserve its internal input graph and PDFs |
| 2 | `write-paper/ACOMPA26_P02/` | `paper/versions/july-rewrite/` | Earlier product; preserve all source and build files |
| 3 | `ACOMPA26_P02/` | `paper/ACOMPA26_P02/` | Active product; move complete folder, preserve relative figure/input links; retain root compatibility symlink |
| 4 | `write-paper/main_proposal6_rewrite_with_mme.md` | `docs/method/main_proposal6_rewrite_with_mme.md` | Historical writing product |
| 5 | `ONLY_report.md` | `docs/method/ONLY_report.md` | Method reference |
| 6–8 | Root `CMD_*_draft.md` | `docs/method/` with identical basenames | Earlier writing products |

The intended active destination already contains the earlier manuscript.
[collisions.csv](collisions.csv) lists the case-folded file collisions. Step 1
must complete and verify before step 3 can populate that destination. Do not
merge these folders or overwrite matching filenames.

Root runner locations stay in place because their `ROOT` derivation, bundle
member names, generated Colab commands, README examples, and tests depend on
them. `ONLY/`, `data/`, `models/`, and `logs/` also stay in place. Root
`colab_bundle.tar.gz` stays a generated transport product at its required path.
Empty `wget-log*` files are held until their consumers are known. The method
and evaluation code is not refactored as part of this organization.

The inventory separates both repositories and records modification time,
size, tracking status, and symlinks. Source/documentation references are
searched in both repositories; downloaded assets, logs, environment folders,
Git internals, protected configuration folders, and files over 2 MB are not
content-scanned. This is a bounded reference audit, not a claim about all
possible host consumers. External consumers such as Overleaf sync, editor
projects, server jobs, or notebooks must be confirmed by the owner before
removing their old entry points. The root paper compatibility link covers the
currently named paper path.

## Catalog and folder contracts to add after review

The root `AGENTS.md` will remain below approximately 60 lines: task → owning
folder, which paper is active, and the sibling-repository distinction. Retain
the provided ICM and Colab instructions. `CONTEXT.md` will describe these steps:

1. **Method:** read `docs/method/` references and the responsible module under
   `ONLY/`; change implementation and relevant tests; review the diff and
   focused checks before an evaluation.
2. **Evaluation:** read the chosen runner, evaluator, and asset contract;
   write raw outputs to existing `logs/` paths. Create an experiment record
   from `_templates/experiment-record.md` naming command, seed, proposal,
   repository revision, uncommitted changes, output paths, and completeness.
   A human verifies coverage and metric definitions before results enter paper
   tables. Existing recovered logs retain their provenance labels.
3. **Manuscript:** read `paper/ACOMPA26_P02/CONTEXT.md` and only the relevant
   included sections and verified evidence. Build from the manuscript folder.
   A human reviews changed text/tables and the compiled output.

Each contract will name concrete input and output paths, the process, and the
human check. Link to authoritative files rather than copying method formulas
or result tables into catalogs. Existing outputs determine available state;
do not label incomplete runs complete or planned ablations performed.

The template is a plain Markdown record, not a new experiment framework.
No empty pipeline stages, speculative dashboards, or duplicate system-map
cards are required for the current workspace.

## Durability, execution, and validation

Both directories are Git repositories. Several current manuscript and result
paths are untracked, and existing root scripts have user changes. Git alone is
not a backup of those files. Before migration, create a dated preservation
bundle of every proposed source and save the original Git status and manifest
inside `docs/reorganization/`; hash and inspect the bundle. Do not commit,
discard, or stash unrelated user changes.

After review of this plan:

1. Re-run `python3 docs/reorganization/audit.py`; reject unexpected collisions,
   duplicate destinations, changed source hashes, or new unknown referrers.
2. Save and verify the preservation bundle, including untracked source files.
3. Copy each move group in the listed order, verify byte hashes and file count,
   then remove only the verified originals. Preserve file modes. Resolve every
   in-scope referrer in the same migration and install the root compatibility
   link. Keep historical snapshots independent and content-identical.
4. Write the catalogs, contracts, experiment-record template, and updated
   README navigation. Leave paper content and section selection intact.
5. Verify every manifest source against its destination; re-check symlinks,
   active and historical LaTeX input/figure paths, and Markdown navigation.
   Run focused existing CLI/mask tests when their dependencies are available.
   A LaTeX build, if tooling is installed, may still expose pre-existing paper
   issues recorded in [paper-readout.md](paper-readout.md); do not silently
   rewrite the paper to make the migration appear clean.
6. Walk cold from `AGENTS.md` to the active paper, Proposal 6 implementation,
   evaluator, and corresponding evidence in no more than two routing reads.
   Record validation and the resulting Git diff; leave reviewable changes
   uncommitted.

## Review decision

The user approved execution by requesting the reorganization be committed and
pushed after this plan was presented. No additional external consumers were
identified; the current paper path retains its compatibility link.
The proposal gate is explicitly required by the invoked ICM skill's
Restructure step 5: “Present the target tree and a migration map ... Get
approval.” The exact instruction is in
[/home/dat/.agents/skills/icm-architect/SKILL.md](/home/dat/.agents/skills/icm-architect/SKILL.md).
