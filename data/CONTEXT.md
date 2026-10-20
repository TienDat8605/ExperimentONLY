# Asset contract

One job: hold the benchmark inputs downloaded or prepared by the existing runners.

Inputs: selected benchmark, model configuration, and asset download command in the root README.
Process: use the existing runner's download/check actions; do not move assets to match a new layout.
Outputs: `coco/val2014/`, `pope/`, `chair/`, and `mme_hallucination/` as required by the evaluator.
Model checkpoints belong at root `models/llava-v1.5-7b/` and
`models/clip-vit-large-patch14-336/` when downloaded, separate from benchmark data.

Human check: confirm paths, split/annotation identity, image coverage, and readiness before inference.
Directory existence alone is not asset completeness. Large downloaded assets remain outside Git;
the current checkout does not need all assets to navigate or edit the paper.
