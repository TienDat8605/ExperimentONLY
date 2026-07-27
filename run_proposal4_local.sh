#!/usr/bin/env bash
# Run Proposal 4 on a local/rented NVIDIA GPU server.
#
# Typical fresh-server workflow:
#   bash run_proposal4_local.sh setup
#   bash run_proposal4_local.sh download
#   bash run_proposal4_local.sh run --short --setups adversarial
#
# Or perform all three stages:
#   bash run_proposal4_local.sh all --short --setups adversarial
#
# The virtual environment inherits the server's PyTorch installation. Install a
# CUDA-enabled PyTorch build in the base Python environment before running setup.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROPOSAL4_VENV:-${ROOT}/.venv-proposal4}"
PYTHON_BIN="${PROPOSAL4_PYTHON:-python3}"
MODEL_DIR="${ROOT}/models/llava-v1.5-7b"
CLIP_DIR="${ROOT}/models/clip-vit-large-patch14-336"
COCO_DIR="${ROOT}/data/coco/val2014"
POPE_DIR="${ROOT}/data/pope"

ACTION="run"
SETUPS="random popular adversarial"
MAX_QUESTIONS=0
MAX_NEW_TOKENS=8
MASK_ALPHA=0.2
JS_GAMMA=0.6
DEBUG_TVD=0
ALPHA_1=""
ALPHA_2=""
ALPHA_3=""
ALPHA_4=""
GAMMA_1=""
GAMMA_2=""

usage() {
    cat <<'EOF'
Usage:
  bash run_proposal4_local.sh setup
  bash run_proposal4_local.sh download
  bash run_proposal4_local.sh check
  bash run_proposal4_local.sh run [options]
  bash run_proposal4_local.sh all [options]

Actions:
  setup       Create .venv-proposal4 and install compatible dependencies.
  download    Download LLaVA-1.5-7B, CLIP, COCO val2014, and POPE.
  check       Validate CUDA, dependencies, model weights, and datasets.
  run         Validate and evaluate Proposal 4 (default action).
  all         Run setup, download, then evaluation.

Run options:
  --setups random popular adversarial   POPE splits to run.
  --setups=adversarial                  One or more quoted split names.
  --short                               Run 300 questions per split.
  --maxq=N                              Run N questions per split (0 = all 3000).
  --tokens=N                            Generated tokens per answer (default: 8).
  --mask-alpha=F                        Mask EMA alpha (Proposal 4 is static).
  --js-gamma=F                          Fallback TVD threshold (default: 0.6).
  --alpha-1=F ... --alpha-4=F           Dual-branch decoding coefficients.
  --gamma-1=F --gamma-2=F               Text and visual TVD thresholds.
  --debug                               Print per-token TVD diagnostics.

Environment:
  PROPOSAL4_PYTHON=/path/to/python      Base Python used to create the venv.
  PROPOSAL4_VENV=/path/to/venv          Override the virtual environment path.
  HF_TOKEN=...                          Optional token for Hugging Face downloads.
  CUDA_VISIBLE_DEVICES=0                Select the GPU used for evaluation.
EOF
}

log() {
    printf '[%(%H:%M:%S)T] %s\n' -1 "$*"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

if [[ $# -gt 0 && "$1" != --* ]]; then
    ACTION="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --setups)
            SETUPS=""
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SETUPS="${SETUPS:+${SETUPS} }$1"
                shift
            done
            ;;
        --setups=*) SETUPS="${1#--setups=}"; shift ;;
        --short) MAX_QUESTIONS=300; shift ;;
        --maxq=*) MAX_QUESTIONS="${1#--maxq=}"; shift ;;
        --tokens=*) MAX_NEW_TOKENS="${1#--tokens=}"; shift ;;
        --mask-alpha=*) MASK_ALPHA="${1#--mask-alpha=}"; shift ;;
        --js-gamma=*) JS_GAMMA="${1#--js-gamma=}"; shift ;;
        --alpha-1=*) ALPHA_1="${1#--alpha-1=}"; shift ;;
        --alpha-2=*) ALPHA_2="${1#--alpha-2=}"; shift ;;
        --alpha-3=*) ALPHA_3="${1#--alpha-3=}"; shift ;;
        --alpha-4=*) ALPHA_4="${1#--alpha-4=}"; shift ;;
        --gamma-1=*) GAMMA_1="${1#--gamma-1=}"; shift ;;
        --gamma-2=*) GAMMA_2="${1#--gamma-2=}"; shift ;;
        --debug) DEBUG_TVD=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

case "$ACTION" in
    setup|download|check|run|all) ;;
    *) usage; die "Unknown action: $ACTION" ;;
esac

venv_python() {
    [[ -x "${VENV_DIR}/bin/python" ]] || die "Virtual environment not found. Run: bash run_proposal4_local.sh setup"
    printf '%s\n' "${VENV_DIR}/bin/python"
}

patch_transformers() {
    local py tf_dir
    py="$(venv_python)"
    tf_dir="$($py -c 'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')"
    cp "${ROOT}/ONLY/patches/modeling_llama.py" "${tf_dir}/models/llama/modeling_llama.py"
    find "${tf_dir}/models/llama" -type d -name __pycache__ -prune -exec rm -rf {} +
    log "Patched ${tf_dir}/models/llama/modeling_llama.py"
}

setup_env() {
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"
    "$PYTHON_BIN" -c 'import torch, torchvision; print(f"Using base PyTorch {torch.__version__}")' || \
        die "Install a CUDA-enabled PyTorch and matching torchvision build from pytorch.org before setup."
    log "Creating isolated environment at ${VENV_DIR}"
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
    local py pip tokenizers_version
    py="${VENV_DIR}/bin/python"
    pip="${VENV_DIR}/bin/pip"
    "$py" -m pip install --upgrade pip wheel setuptools

    if "$py" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
        tokenizers_version="0.19.1"
    else
        tokenizers_version="0.13.3"
    fi

    "$pip" install \
        accelerate==0.21.0 sentencepiece einops==0.6.1 timm==0.6.13 \
        scipy opencv-python pycocotools pandas pyarrow pillow tqdm pyyaml \
        requests regex packaging safetensors
    "$pip" install --no-deps \
        transformers==4.31.0 huggingface_hub==0.16.4 "tokenizers==${tokenizers_version}"

    # Python 3.12 has no tokenizers 0.13 wheel. The repository already uses the
    # compatible 0.19 wheel on Colab; disable only Transformers' strict version
    # gate while retaining the pinned Transformers API expected by the patch.
    if [[ "$tokenizers_version" == "0.19.1" ]]; then
        "$py" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("transformers")
path = Path(spec.origin).parent / "dependency_versions_check.py"
text = path.read_text()
text = text.replace("require_version_core(deps[pkg])", "pass  # ONLY: tokenizers 0.19 on Python 3.12")
path.write_text(text)
PY
    fi

    "$py" -c 'import torch; print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")' || \
        die "PyTorch is missing. Install the CUDA build recommended by your GPU provider, then rerun setup."
    patch_transformers
    log "Environment setup complete"
}

download_assets() {
    local py
    py="$(venv_python)"
    mkdir -p "${ROOT}/models" "${ROOT}/data/coco" "$POPE_DIR"

    if [[ ! -s "${MODEL_DIR}/config.json" ]]; then
        log "Downloading LLaVA-1.5-7B (about 14 GB)"
        MODEL_DIR="$MODEL_DIR" "$py" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("liuhaotian/llava-v1.5-7b", local_dir=os.environ["MODEL_DIR"], local_dir_use_symlinks=False, resume_download=True)
PY
    else
        log "LLaVA weights already present"
    fi

    if [[ ! -s "${CLIP_DIR}/config.json" ]]; then
        log "Downloading CLIP ViT-L/14-336 (about 1.7 GB)"
        CLIP_DIR="$CLIP_DIR" "$py" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("openai/clip-vit-large-patch14-336", local_dir=os.environ["CLIP_DIR"], local_dir_use_symlinks=False, resume_download=True)
PY
    else
        log "CLIP weights already present"
    fi

    if [[ ! -d "$COCO_DIR" ]] || [[ -z "$(find "$COCO_DIR" -maxdepth 1 -name '*.jpg' -print -quit 2>/dev/null)" ]]; then
        command -v wget >/dev/null 2>&1 || die "wget is required to download COCO"
        command -v unzip >/dev/null 2>&1 || die "unzip is required to extract COCO"
        log "Downloading COCO val2014 (about 4 GB)"
        local archive="${ROOT}/data/coco/val2014.zip"
        wget -c https://images.cocodataset.org/zips/val2014.zip -O "$archive"
        unzip -tq "$archive" >/dev/null
        unzip -q "$archive" -d "${ROOT}/data/coco"
        rm -f "$archive"
    else
        log "COCO val2014 already present"
    fi

    if [[ ! -s "${POPE_DIR}/coco_pope_adversarial.json" ]]; then
        log "Downloading and converting POPE annotations"
        POPE_DIR="$POPE_DIR" "$py" - <<'PY'
import json
import os
import pandas as pd
from huggingface_hub import hf_hub_download

out_dir = os.environ["POPE_DIR"]
splits = {
    "random": "data/test-00000-of-00003.parquet",
    "popular": "data/test-00001-of-00003.parquet",
    "adversarial": "data/test-00002-of-00003.parquet",
}
for split, filename in splits.items():
    source = hf_hub_download("lmms-lab/POPE", filename=filename, repo_type="dataset")
    frame = pd.read_parquet(source)
    target = os.path.join(out_dir, f"coco_pope_{split}.json")
    with open(target, "w", encoding="utf-8") as handle:
        for _, row in frame.iterrows():
            answer = str(row.answer)
            record = {
                "question_id": int(row.question_id),
                "image": str(row.image_source) + ".jpg",
                "text": str(row.question),
                "answer": answer,
                "label": answer,
            }
            handle.write(json.dumps(record) + "\n")
    print(f"{split}: {len(frame)} questions")
PY
    else
        log "POPE annotations already present"
    fi
}

check_ready() {
    local py failed=0
    py="$(venv_python)"
    "$py" - <<'PY' || failed=1
import sys
import torch
import transformers

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.total_memory / 2**30:.1f} GiB)")
else:
    raise SystemExit("A CUDA-capable NVIDIA GPU is required")
PY

    for required in \
        "${MODEL_DIR}/config.json" \
        "${CLIP_DIR}/config.json" \
        "${POPE_DIR}/coco_pope_random.json" \
        "${POPE_DIR}/coco_pope_popular.json" \
        "${POPE_DIR}/coco_pope_adversarial.json"; do
        if [[ ! -s "$required" ]]; then
            echo "Missing: $required" >&2
            failed=1
        fi
    done
    if [[ ! -d "$COCO_DIR" ]] || [[ -z "$(find "$COCO_DIR" -maxdepth 1 -name '*.jpg' -print -quit 2>/dev/null)" ]]; then
        echo "Missing COCO images: $COCO_DIR" >&2
        failed=1
    fi
    [[ "$failed" -eq 0 ]] || die "Prerequisite check failed. Run the setup/download actions or provide the missing assets."
    log "All Proposal 4 prerequisites are ready"
}

run_eval() {
    check_ready
    patch_transformers

    [[ "$MAX_QUESTIONS" =~ ^[0-9]+$ ]] || die "--maxq must be a non-negative integer"
    [[ "$MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "--tokens must be a positive integer"
    for setup in $SETUPS; do
        case "$setup" in random|popular|adversarial) ;; *) die "Unknown POPE setup: $setup" ;; esac
    done

    local py run_dir overall_exit=0
    py="$(venv_python)"
    run_dir="${ROOT}/logs/results_$(date +%Y-%m-%d_%Hh%Mm%Ss)_proposal4_local"
    mkdir -p "$run_dir"
    export PYTHONPATH="${ROOT}/ONLY:${ROOT}/ONLY/eval_bench:${ROOT}/ONLY/experiments${PYTHONPATH:+:${PYTHONPATH}}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
    export RANK=0
    export WORLD_SIZE=1

    for setup in $SETUPS; do
        local output_file="${run_dir}/proposal4_result_${setup}.txt"
        local extra_args=()
        [[ -n "$ALPHA_1" ]] && extra_args+=(--alpha_1 "$ALPHA_1")
        [[ -n "$ALPHA_2" ]] && extra_args+=(--alpha_2 "$ALPHA_2")
        [[ -n "$ALPHA_3" ]] && extra_args+=(--alpha_3 "$ALPHA_3")
        [[ -n "$ALPHA_4" ]] && extra_args+=(--alpha_4 "$ALPHA_4")
        [[ -n "$GAMMA_1" ]] && extra_args+=(--gamma_1 "$GAMMA_1")
        [[ -n "$GAMMA_2" ]] && extra_args+=(--gamma_2 "$GAMMA_2")

        log "Running Proposal 4 on POPE/${setup}; output: ${output_file}"
        set +e
        "$py" -u "${ROOT}/ONLY/eval_bench/pope_eval_llava.py" \
            --model_path "$MODEL_DIR" \
            --model_base llava \
            --pope_path "${POPE_DIR}/coco_pope_${setup}.json" \
            --data_path "$COCO_DIR" \
            --log_path "${ROOT}/logs" \
            --use_only True \
            --type "$setup" \
            --dataset_name coco \
            --enhance_layer_index 0 \
            --mask_alpha "$MASK_ALPHA" \
            --temperature 1.0 \
            --top_p 1 \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --max_questions "$MAX_QUESTIONS" \
            --debug_tvd "$([[ "$DEBUG_TVD" -eq 1 ]] && echo True || echo False)" \
            --proposal 4 \
            --js_gamma "$JS_GAMMA" \
            --batch_size 1 \
            --num_workers 1 \
            --seed 42 \
            "${extra_args[@]}" \
            2>&1 | tee "$output_file"
        local eval_exit=${PIPESTATUS[0]}
        set -e
        if [[ "$eval_exit" -ne 0 ]]; then
            overall_exit="$eval_exit"
            log "POPE/${setup} failed with exit code ${eval_exit}"
        else
            log "POPE/${setup} completed"
            grep 'acc:' "$output_file" | tail -1 || true
        fi
    done

    log "Results directory: ${run_dir}"
    return "$overall_exit"
}

cd "$ROOT"
case "$ACTION" in
    setup) setup_env ;;
    download) download_assets ;;
    check) check_ready ;;
    run) run_eval ;;
    all)
        setup_env
        download_assets
        run_eval
        ;;
esac
