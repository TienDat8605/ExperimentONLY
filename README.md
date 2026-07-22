# ONLY — One-Layer Intervention Sufficiently Mitigates Hallucinations

A training-free decoding method for reducing hallucinations in Vision-Language Models (VLMs). Suppresses "entropy-imbalanced" attention heads to produce a contrastive logit distribution, then combines it with the normal distribution via a TVD-based switching rule.

## Method

ONLY produces two logit distributions in a single forward pass:

- **Normal logits** — standard model output
- **Contrastive logits** — computed from a "head-suppressed" hidden state (`hidden_states_cd`) threaded through all 32 layers

At the **enhance layer** (default: layer 0), attention heads with low text-entropy / high image-entropy ratio are zeroed out. This ghost signal passes untouched through layers 1–30, gets full decoder processing at layer 31, and is combined with the normal output via a TVD switching rule at each generation step.

## Proposals

| Proposal | Description |
|----------|-------------|
| 0 | **Original ONLY** — binary mask computed once at layer 0 only |
| 1 | **EMA mask accumulation** — fresh binary mask per layer, blended via EMA (`alpha=0.2`) |
| 2 | **Score accumulation** — track continuous score per head (ratio - threshold), sigmoid-activated |
| 3 | **Residual delta tracking** — binary mask per layer, accumulate `delta = masked - normal` attn output with scaling (`lambda_decay=0.3`) |
| 4 | **Layer-Local Consensus ONLY** (default) — independent soft-mask experts at layers 0/8/16/24, combined in vocabulary space with layer-0 ONLY fallback |

## Project Structure

```
├── colab.sh                 # Prep (download) + run (eval on Colab/local)
├── run_all.sh               # Orchestrator: local → Colab upload → eval → download
├── colab_runner.py          # Python wrapper launched by run_all.sh on Colab
├── colab_bundle.tar.gz      # Code tarball uploaded to Colab VM
├── recover_pope_log.py      # Recover results from Colab CLI history if download fails
├── ONLY/
│   ├── patches/modeling_llama.py     # Patched LLaMA attention with ONLY logic
│   ├── eval_bench/pope_eval_llava.py  # POPE evaluation entry point
│   ├── only_utils/only_sample.py      # Custom sampling with TVD switching
│   ├── experiments/llava/             # LLaVA model files
│   └── utils/                         # Logging & distributed utils
├── models/
│   ├── llava-v1.5-7b/                 # LLaVA model weights
│   └── clip-vit-large-patch14-336/    # CLIP vision encoder
└── data/
    ├── coco/val2014/                  # COCO validation images
    └── pope/                          # POPE evaluation data (random/popular/adversarial)
```

## Requirements

- Python 3.12
- PyTorch, transformers, huggingface_hub
- GPU with ~12 GB VRAM (T4 works)
- ~20 GB disk for model weights + data

## Usage

### Local Run

```bash
# 1. Download assets (models + data)
bash colab.sh prep

# 2. Run evaluation locally
bash colab.sh run
```

### Colab Run (recommended for GPU)

```bash
# 1. Create the code tarball
bash colab.sh prep

# 2. Upload colab_bundle.tar.gz to Colab, then:
#    (on Colab) bash colab.sh run

# Or use the automated runner (requires colab CLI):
bash run_all.sh
```

### Command-Line Options

```bash
# Run with specific proposal
bash run_all.sh --proposal=4              # Layer-local consensus (default)
bash run_all.sh --proposal=1              # EMA mask ablation
bash run_all.sh --proposal=0              # Original ONLY (layer 0 only)
bash run_all.sh --proposal=2              # Score accumulation
bash run_all.sh --proposal=3              # Residual delta tracking
bash run_all.sh --proposal=4 --expert_layers=0,8,16,24 \
  --consensus_min=0.75 --consensus_strength=1.0

# Evaluation flags
bash run_all.sh --short                   # 300 questions per setup (quick test)
bash run_all.sh --setups random popular   # Only these POPE setups
bash run_all.sh --debug                   # Verbose per-step TVD logs

# Proposal-specific tuning
bash run_all.sh --proposal=2 --score_threshold=0.0 --score_temperature=1.0
bash run_all.sh --proposal=3 --lambda_decay=0.3

# Session management
bash run_all.sh --fresh                  # Force new Colab VM
bash run_all.sh --stop                   # Stop VM after run
bash run_all.sh --tokens=8              # max_new_tokens (default 8)
bash run_all.sh --alpha=-0.2            # adaptive alpha for proposal 1 (<0 = per-step dynamic)
```

### Local Eval (without Colab)

```bash
export POPE_PROPOSAL=4
export POPE_ALPHA=0.2
bash colab.sh run
```

## POPE Evaluation

Evaluates on three POPE (Polling-based Object Probing Evaluation) setups:

- **random** — random negative objects
- **popular** — frequently co-occurring negative objects (harder)
- **adversarial** — adversarially selected negative objects (hardest)

Outputs accuracy, precision, recall, F1 per setup to `logs/results_<timestamp>/`.
POPE uses the paper's LLaVA-1.5 threshold `gamma=0.2`. CHAIR uses 500
seeded-random images, `gamma=0.25`, and the scorer/cache released by the ONLY
authors; the captions-only score printed during generation is diagnostic only.

## Patch System

The ONLY attention logic patches `transformers.models.llama.modeling_llama.py` at runtime. The patched version is at `ONLY/patches/modeling_llama.py`. The `colab.sh run` step copies it over the installed transformers version.

## Reference

> ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations
> Paper: [https://arxiv.org/abs/2507.00898](https://arxiv.org/abs/2507.00898)
