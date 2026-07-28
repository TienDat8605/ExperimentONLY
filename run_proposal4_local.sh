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
# The setup action creates a self-contained virtual environment and installs the
# tested CUDA 12.8 PyTorch build. The host only needs a compatible NVIDIA driver.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROPOSAL4_VENV:-${ROOT}/.venv-proposal4}"
PYTHON_BIN="${PROPOSAL4_PYTHON:-python3}"
MODEL_DIR="${ROOT}/models/llava-v1.5-7b"
CLIP_DIR="${ROOT}/models/clip-vit-large-patch14-336"
COCO_DIR="${ROOT}/data/coco/val2014"
POPE_DIR="${ROOT}/data/pope"
TORCH_VERSION="${PROPOSAL4_TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${PROPOSAL4_TORCHVISION_VERSION:-0.26.0}"
PYTORCH_INDEX_URL="${PROPOSAL4_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
COCO_BASE_URL="${PROPOSAL4_COCO_BASE_URL:-http://images.cocodataset.org/val2014}"
COCO_WORKERS="${PROPOSAL4_COCO_WORKERS:-16}"
HF_DOWNLOAD_TIMEOUT="${PROPOSAL4_HF_TIMEOUT:-120}"

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
  download    Download models, POPE, and only the COCO images POPE needs.
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
  PROPOSAL4_PYTHON=/path/to/python      Python used to create the venv.
  PROPOSAL4_VENV=/path/to/venv          Override the virtual environment path.
  PROPOSAL4_COCO_WORKERS=16             Concurrent COCO image downloads.
  PROPOSAL4_COCO_BASE_URL=http://...    Override the COCO image endpoint.
  PROPOSAL4_HF_TIMEOUT=120              Hugging Face download timeout in seconds.
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

transformers_dir() {
    local py
    py="$(venv_python)"
    "$py" -c 'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "transformers")'
}

patch_transformers_version_gate() {
    local py tf_dir
    py="$(venv_python)"
    tf_dir="$(transformers_dir)"
    TF_DIR="$tf_dir" "$py" - <<'PY'
import os
import re
from pathlib import Path

root = Path(os.environ["TF_DIR"])
check_path = root / "dependency_versions_check.py"
table_path = root / "dependency_versions_table.py"
if not check_path.is_file() or not table_path.is_file():
    raise SystemExit(f"Transformers is not installed correctly at {root}")

# Repair files produced by the older non-idempotent workaround. Every inserted
# guard is removed and the official require_version_core call is restored at
# its original indentation. This is safe to run repeatedly.
lines = check_path.read_text().splitlines()
repaired = []
found_requirement = False
for line in lines:
    if 'if pkg != "tokenizers":  # ONLY:' in line:
        continue
    if "pass  # ONLY: tokenizers 0.19" in line:
        line = "        require_version_core(deps[pkg])"
    elif "require_version_core(deps[pkg])" in line:
        line = "        require_version_core(deps[pkg])"
    if "require_version_core(deps[pkg])" in line:
        if found_requirement:
            continue
        found_requirement = True
    repaired.append(line)
if not found_requirement:
    raise SystemExit(f"Could not repair the dependency check in {check_path}")
check_text = "\n".join(repaired) + "\n"
compile(check_text, str(check_path), "exec")
check_path.write_text(check_text)

# Transformers 4.31 pins tokenizers below 0.14, whose wheels do not support
# Python 3.12. Loosen only that dependency entry; keep every other version gate.
table_text = table_path.read_text()
pattern = r'("tokenizers"\s*:\s*"tokenizers[^"\n]*<)0\.14(")'
updated, count = re.subn(pattern, r"\g<1>0.20\2", table_text)
if count == 0 and not re.search(r'"tokenizers"\s*:\s*"tokenizers[^"\n]*<0\.20"', table_text):
    raise SystemExit(f"Could not adjust the tokenizers requirement in {table_path}")
table_path.write_text(updated)
PY
}

patch_transformers() {
    local tf_dir
    patch_transformers_version_gate
    tf_dir="$(transformers_dir)"
    cp "${ROOT}/ONLY/patches/modeling_llama.py" "${tf_dir}/models/llama/modeling_llama.py"
    find "${tf_dir}/models/llama" -type d -name __pycache__ -prune -exec rm -rf {} +
    log "Patched ${tf_dir}/models/llama/modeling_llama.py"
}

setup_env() {
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"
    log "Creating isolated environment at ${VENV_DIR}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    local py pip tokenizers_version
    py="${VENV_DIR}/bin/python"
    pip="${VENV_DIR}/bin/pip"
    "$py" -m pip install --upgrade pip wheel setuptools

    log "Installing PyTorch ${TORCH_VERSION} with CUDA 12.8"
    "$pip" install \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "$PYTORCH_INDEX_URL"

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
        patch_transformers_version_gate
    fi

    "$py" -c 'import torch, torchvision; print(f"PyTorch {torch.__version__}; torchvision {torchvision.__version__}; runtime CUDA {torch.version.cuda}")' || \
        die "The CUDA 12.8 PyTorch installation failed."
    patch_transformers
    log "Environment setup complete"
}

download_pope() {
    local py
    py="$(venv_python)"
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
}

download_coco_subset() {
    local py
    py="$(venv_python)"
    log "Downloading only the unique COCO images referenced by POPE (${COCO_WORKERS} workers)"
    POPE_DIR="$POPE_DIR" COCO_DIR="$COCO_DIR" COCO_BASE_URL="$COCO_BASE_URL" COCO_WORKERS="$COCO_WORKERS" "$py" - <<'PY'
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

pope_dir = Path(os.environ["POPE_DIR"])
coco_dir = Path(os.environ["COCO_DIR"])
base_url = os.environ["COCO_BASE_URL"].rstrip("/")
workers = int(os.environ["COCO_WORKERS"])
if workers < 1:
    raise SystemExit("PROPOSAL4_COCO_WORKERS must be a positive integer")

coco_dir.mkdir(parents=True, exist_ok=True)
names = set()
for annotation in sorted(pope_dir.glob("coco_pope_*.json")):
    with annotation.open(encoding="utf-8") as handle:
        for line in handle:
            names.add(Path(json.loads(line)["image"]).name)
if not names:
    raise SystemExit(f"No POPE image references found in {pope_dir}")

def valid_image(path):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False

pending = [name for name in sorted(names) if not valid_image(coco_dir / name)]
print(f"COCO images: {len(names)} required, {len(names) - len(pending)} present, {len(pending)} to download")

def download(name):
    target = coco_dir / name
    partial = target.with_suffix(target.suffix + ".part")
    url = f"{base_url}/{name}"
    for attempt in range(1, 6):
        try:
            with requests.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if not valid_image(partial):
                raise RuntimeError("downloaded file is not a valid image")
            partial.replace(target)
            return name
        except Exception as exc:
            partial.unlink(missing_ok=True)
            if attempt == 5:
                raise RuntimeError(f"{name}: {exc}") from exc
            time.sleep(attempt * 2)

failures = []
completed = 0
with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = {executor.submit(download, name): name for name in pending}
    for future in as_completed(futures):
        completed += 1
        try:
            future.result()
        except Exception as exc:
            failures.append(str(exc))
        if completed % 100 == 0 or completed == len(pending):
            print(f"Downloaded {completed}/{len(pending)}", flush=True)

if failures:
    print("COCO download failures:")
    for failure in failures[:20]:
        print(f"  {failure}")
    raise SystemExit(f"Failed to download {len(failures)} COCO images; rerun to retry")
PY
}

download_model_snapshot() {
    local repo_id="$1" target_dir="$2" label="$3" py
    py="$(venv_python)"
    mkdir -p "$target_dir"
    REPO_ID="$repo_id" TARGET_DIR="$target_dir" MODEL_LABEL="$label" \
        HF_HUB_ENABLE_HF_TRANSFER=0 \
        HF_HUB_DOWNLOAD_TIMEOUT="$HF_DOWNLOAD_TIMEOUT" \
        HF_HUB_ETAG_TIMEOUT="$HF_DOWNLOAD_TIMEOUT" \
        "$py" -u - <<'PY'
import os
import threading
import time
from pathlib import Path

repo_id = os.environ["REPO_ID"]
target_dir = Path(os.environ["TARGET_DIR"])
label = os.environ["MODEL_LABEL"]

def directory_size(path):
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    value = float(total)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024

def heartbeat(stop):
    while not stop.wait(30):
        print(f"  {label} download is active; local size: {directory_size(target_dir)}", flush=True)

last_error = None
for attempt in range(1, 4):
    print(f"  {label} download attempt {attempt}/3", flush=True)
    stop = threading.Event()
    worker = threading.Thread(target=heartbeat, args=(stop,), daemon=True)
    worker.start()
    try:
        # Import after setting the transfer/time-out environment variables.
        # hf_transfer is deliberately disabled: its legacy path can stall
        # without progress or resumable error handling on rented servers.
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"  {label} synchronized ({directory_size(target_dir)})", flush=True)
        last_error = None
        break
    except Exception as exc:
        last_error = exc
        print(f"  {label} attempt {attempt} failed: {exc!r}", flush=True)
    finally:
        stop.set()
        worker.join(timeout=1)
    if attempt < 3:
        delay = attempt * 15
        print(f"  Retrying in {delay}s; completed files will be reused", flush=True)
        time.sleep(delay)

if last_error is not None:
    raise SystemExit(f"{label} download failed after 3 attempts: {last_error!r}")
PY
}

download_assets() {
    mkdir -p "${ROOT}/models" "${ROOT}/data/coco" "$POPE_DIR"

    log "Synchronizing LLaVA-1.5-7B (about 14 GB; resumable)"
    download_model_snapshot "liuhaotian/llava-v1.5-7b" "$MODEL_DIR" "LLaVA-1.5-7B"

    log "Synchronizing CLIP ViT-L/14-336 (about 1.7 GB; resumable)"
    download_model_snapshot "openai/clip-vit-large-patch14-336" "$CLIP_DIR" "CLIP ViT-L/14-336"

    download_pope
    download_coco_subset
}

check_ready() {
    local py failed=0
    py="$(venv_python)"
    patch_transformers
    "$py" - <<'PY' || failed=1
import sys
import torch
import torchvision
import transformers
import tokenizers

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"Transformers: {transformers.__version__}")
print(f"tokenizers: {tokenizers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"Expected the CUDA 12.8 PyTorch build, found {torch.version.cuda}")
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
    if ! MODEL_DIR="$MODEL_DIR" CLIP_DIR="$CLIP_DIR" POPE_DIR="$POPE_DIR" COCO_DIR="$COCO_DIR" "$py" - <<'PY'
import json
import os
from pathlib import Path

model_dir = Path(os.environ["MODEL_DIR"])
clip_dir = Path(os.environ["CLIP_DIR"])
pope_dir = Path(os.environ["POPE_DIR"])
coco_dir = Path(os.environ["COCO_DIR"])

def weight_files(directory):
    return list(directory.glob("*.bin")) + list(directory.glob("*.safetensors"))

errors = []
if not weight_files(model_dir):
    errors.append(f"No LLaVA weight files found in {model_dir}")
if not weight_files(clip_dir):
    errors.append(f"No CLIP weight files found in {clip_dir}")

expected = set()
for split in ("random", "popular", "adversarial"):
    annotation = pope_dir / f"coco_pope_{split}.json"
    if not annotation.is_file():
        continue
    with annotation.open(encoding="utf-8") as handle:
        for line in handle:
            expected.add(Path(json.loads(line)["image"]).name)

missing = sorted(name for name in expected if not (coco_dir / name).is_file())
print(f"POPE COCO images: {len(expected) - len(missing)}/{len(expected)} present")
if not expected:
    errors.append(f"No COCO image references found in {pope_dir}")
if missing:
    errors.append(f"Missing {len(missing)} POPE COCO images (first: {missing[:5]})")

if errors:
    raise SystemExit("\n".join(errors))
PY
    then
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
