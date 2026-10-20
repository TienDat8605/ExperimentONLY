---
type: manuscript-orientation
status: source-read
---

# ACOMPA26_P02 source readout

This is an organizational readout of local source, not a peer review or a
verification of external bibliography metadata or experimental results.

## Active input graph

The user identified the root `ACOMPA26_P02/` as the paper. Its
[`main.tex`](../../ACOMPA26_P02/main.tex) uses IEEEtran conference format and
includes, in order, `abstract_`, `introduction_`, `relatedwork_`, `method_`,
`experiment_`, `conclusion_`, and `bibliography_`. Those are the active editing
surfaces. The unsuffixed sections and `introduction(2)_.tex` remain alternate
drafts and must not be selected automatically during organization.

`relatedwork_.tex` includes `visual.png` relative to the manuscript directory.
The bibliography is inline LaTeX, not a `.bib` database. Preserve the complete
directory as one unit so these relative references continue to work.

Two earlier versions live in `paper/ACOMPA26_P02/` and
`write-paper/ACOMPA26_P02/`; neither is the root manuscript's current entry
point. The sibling repository also has its own `paper/ACOMPA26_P02/` snapshot.

## Paper-to-code routing

The paper calls Proposal 6 **Adaptive Dual-Branch Decoding (ADBD)**:

| Paper component | Authoritative implementation |
|---|---|
| Sigmoid of centered, population-standardized TVER | `ONLY/only_utils/proposal6_mask.py:continuous_tver_evidence` |
| Confidence × innovation gain and complementary state | `ONLY/only_utils/proposal6_mask.py:update_proposal6_mask` |
| Disjoint high/low-TVER masks | `ONLY/only_utils/proposal6_mask.py:proposal6_keep_masks` |
| Exact update-layer set, final layer reserved | `ONLY/only_utils/proposal6_mask.py:resolve_proposal6_layers` |
| Attention masks and dual ghost-path integration | `ONLY/patches/modeling_llama.py` |
| Independent branch TVD and anchored logit combination | `ONLY/only_utils/only_sample.py:sample` |
| Model parameter forwarding and auxiliary logits | `ONLY/experiments/llava/model/language_model/llava_llama.py` |
| POPE configuration and metrics | `ONLY/eval_bench/pope_eval_llava.py` |
| Local setup, layer options, execution | `run_proposal4_local.sh` (also supports Proposal 6) |

Mask accumulation is across selected decoder layers within a forward call;
the state is initialized again on the next forward call. Branch labels in code
(`vis`, `txt`) should not be treated as experimentally proven specialization.
The paper uses the more descriptive high/low-TVER names.

## Items to keep visible during organization

- `experiment_.tex` reports ADBD POPE accuracy 82.10 / 86.37 / 89.13 and macro
  accuracy 85.87 versus ONLY 85.03. It also reports macro F1 85.48 versus ONLY
  85.49. The prose's broad improvement claims and `(TBD)` markers need a later
  evidence-based writing pass; organization does not resolve them.
- ADBD MME categories 180.00 + 121.67 + 153.33 + 175.00 sum to **630.00**,
  while the table and conclusion say **620.00**. ONLY's displayed categories
  sum to **635.54**, versus displayed total **635.55**, potentially a rounding
  difference. Preserve reported values until raw result provenance is checked.
- `experiment_.tex` references `tab:detailed_results`, but the active input
  graph does not define that label. A more extensive table exists in an
  alternate draft; do not switch drafts to hide the missing reference.
- Ablations, state/gain analysis, branch dynamics, and efficiency measurements
  are described or planned without completed evidence tables in the active
  source. The conclusion already attributes gains to ablations; their
  completion needs verification before submission.
- The paper says MME uses the official category protocol. This repository's
  `mme_hallucination_eval_llava.py:compute_metrics_by_category` reports ordinary
  classification metrics; the exact source of the paper's aggregate MME
  scores must be identified. The sibling evaluator is a distinct surface.
- The current root manuscript is untracked. Preserve it explicitly before
  relocating it; do not rely on the tracked earlier paper as its backup.

Static input, figure, reference-label, and citation-key checks are generated
in [audit-summary.json](audit-summary.json). They do not establish successful
typesetting, citation accuracy, or experiment reproducibility.
