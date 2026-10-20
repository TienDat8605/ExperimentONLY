# Benchmark contract

One job: generate and score predictions for a selected benchmark and decoding configuration.

## Inputs

- `pope_eval_llava.py` and `pope_loader.py`: POPE splits, labels, generation, classification metrics.
- `chair_eval_llava.py`: caption generation and this checkout's CHAIR calculation.
- `mme_hallucination_eval_llava.py`: MME questions and this checkout's category metrics.
- Root `run_proposal4_local.sh`: local setup/download/check/run, including Proposal 6 layer/gain options.
- Root `colab.sh`, `run_all.sh`, `run_proposal4_colab.sh`: existing remote execution entry points.
- [../../data/CONTEXT.md](../../data/CONTEXT.md) and [../CONTEXT.md](../CONTEXT.md).

## Process and outputs

Inspect the chosen entry point's argument support; support is not identical across benchmarks.
Run with the exact intended model, split, seed, token budget, proposal, and thresholds.
Keep raw predictions and metric logs under the runner/evaluator's existing `logs/` locations.
Use [../../logs/CONTEXT.md](../../logs/CONTEXT.md) to record the run and its provenance.

## Human check

Confirm full versus capped question counts and successful completion before comparing results.
Verify metric definitions in code; ordinary MME accuracy/F1 does not establish official MME
aggregate scores, and caption diagnostics do not establish official CHAIR results.
Do not import results or scoring assumptions from the sibling evaluator without an explicit check.
