# Implementation contract

One job: implement inference-time interventions and the model plumbing used by evaluation.

## Inputs

- `patches/modeling_llama.py`: runtime replacement for Transformers' LLaMA module.
- `only_utils/proposal6_mask.py`: continuous evidence, gains, complementary masks, selected layers.
- `only_utils/only_sample.py`: sampling, branch disagreement, and anchored logits.
- `experiments/llava/model/language_model/llava_llama.py`: custom arguments and auxiliary logits.
- Selected evaluator under `eval_bench/`; root runners install/activate the patch.

## Process and outputs

Follow an option from runner → evaluator → model wrapper → LLaMA → sampler before changing it.
Outputs are source edits and focused verification. Preserve the vendored model interfaces and
the pinned Transformers workflow. Changes to mask semantics hit mask tests, the patch, and the
paper's method section; changes to decoding hit the sampler and the paper's resolution equations.
No result table is updated automatically by a code change.

## Human check

Review the affected interfaces and targeted checks. Check the actual proposal branch rather than
assuming proposal IDs have the same meaning in the sibling repository.
For evaluation, continue at [eval_bench/CONTEXT.md](eval_bench/CONTEXT.md).
