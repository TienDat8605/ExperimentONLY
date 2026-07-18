#!/bin/bash
# Proposal 1: Damped Mask Accumulation for ONLY
#
# Usage:
#   bash colab.sh prep     # local: download + tarball
#   bash colab.sh run      # Colab: unpack + eval
#
# "prep" downloads missing assets and creates colab_bundle.tar.gz.
# Upload that tarball to Colab, then run "bash colab.sh run" there.
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${ROOT}/models/llava-v1.5-7b"
CLIP_DIR="${ROOT}/models/clip-vit-large-patch14-336"
COCO_DIR="${ROOT}/data/coco/val2014"
POPE_DIR="${ROOT}/data/pope"
TARBALL="${ROOT}/colab_bundle.tar.gz"

ENV="${ENV:-mymethod}"  # override via ENV=xxx bash colab.sh prep

dbg() {
    echo "[$(date '+%H:%M:%S')] $*"
}

prep() {
    dbg "=== [prep] Download missing assets ==="
    cd "$ROOT"

    # CLIP
    dbg "[prep/1] Checking CLIP..."
    if [ ! -f "$CLIP_DIR/config.json" ]; then
        dbg "[prep/1] Downloading CLIP..."
        mkdir -p "$CLIP_DIR"
        conda run -n "$ENV" python -c "
import os; os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import snapshot_download
snapshot_download('openai/clip-vit-large-patch14-336', local_dir='$CLIP_DIR',
                  local_dir_use_symlinks=False, resume_download=True)
" >/dev/null 2>&1
        dbg "[prep/1] CLIP download complete"
    else
        dbg "[prep/1] CLIP exists, skipping"
    fi

    # COCO val2014
    dbg "[prep/2] Checking COCO val2014..."
    if [ ! -d "$COCO_DIR" ]; then
        # Ensure clean slate: remove partial/corrupt download
        rm -f /tmp/val2014.zip
        dbg "[prep/2] Downloading COCO val2014 (~4 GB, may take a while)..."
        mkdir -p "${ROOT}/data/coco"
        dbg "[prep/2] wget -c http://images.cocodataset.org/zips/val2014.zip → /tmp/val2014.zip"
        wget -c -q http://images.cocodataset.org/zips/val2014.zip -O /tmp/val2014.zip 2>/dev/null
        dbg "[prep/2] Downloaded. Checking zip integrity..."
        if ! unzip -tq /tmp/val2014.zip >/dev/null 2>&1; then
            dbg "[prep/2] ⚠️  Zip corrupt. Removing and retrying..."
            rm -f /tmp/val2014.zip
            wget -c -q http://images.cocodataset.org/zips/val2014.zip -O /tmp/val2014.zip 2>/dev/null
            dbg "[prep/2] Retry done. Checking integrity again..."
            unzip -tq /tmp/val2014.zip >/dev/null 2>&1
        fi
        dbg "[prep/2] Zip OK. Extracting to ${ROOT}/data/coco/..."
        unzip -q /tmp/val2014.zip -d "${ROOT}/data/coco/"
        rm /tmp/val2014.zip
        IMG_COUNT=$(ls "$COCO_DIR" | wc -l)
        dbg "[prep/2] COCO done: ${IMG_COUNT} images"
    else
        IMG_COUNT=$(ls "$COCO_DIR" | wc -l)
        dbg "[prep/2] COCO exists (${IMG_COUNT} images), skipping"
    fi

    # POPE annotations
    dbg "[prep/3] Checking POPE annotations..."
    if [ ! -f "${POPE_DIR}/coco_pope_adversarial.json" ]; then
        dbg "[prep/3] POPE annotations missing, re-downloading..."
        rm -rf "$POPE_DIR"
        mkdir -p "$POPE_DIR"
        dbg "[prep/3] Ensuring pandas + huggingface_hub in conda env..."
        conda run -n "$ENV" pip install -q pandas huggingface_hub pyarrow 2>&1 | tail -1
        dbg "[prep/3] Fetching parquet files from lmms-lab/POPE..."
        conda run -n "$ENV" python -c "
import pandas as pd, json
from huggingface_hub import hf_hub_download
for tp, pp in [('random','data/test-00000-of-00003.parquet'), ('popular','data/test-00001-of-00003.parquet'), ('adversarial','data/test-00002-of-00003.parquet')]:
    print(f'  Downloading {pp}...', flush=True)
    lp = hf_hub_download(repo_id='lmms-lab/POPE', filename=pp, repo_type='dataset')
    df = pd.read_parquet(lp)
    # Write JSON Lines (one record per line) — pope_loader.py reads the file
    # line-by-line and does json.loads(q) on each, expecting a dict per line.
    # (json.dump(records) wrote a single-line array, which made json.loads
    # return a list and broke line['image'].)
    # label must be the answer STRING ('yes'/'no'), NOT a pre-converted int —
    # pope_loader.py normalizes via `if label == 'no': 0 else 1`; an int would
    # always hit the else-branch and label every question as 'yes'.
    records = [{'question_id': int(r.question_id), 'image': str(r.image_source) + '.jpg', 'text': str(r.question), 'answer': str(r.answer), 'label': str(r.answer)} for _, r in df.iterrows()]
    with open('$POPE_DIR/coco_pope_' + tp + '.json', 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
    print(f'  POPE {tp}: {len(records)} records (JSONL)', flush=True)
"
        for f in "$POPE_DIR"/coco_pope_*.json; do
            dbg "[prep/3]  $(basename "$f"): $(wc -c < "$f") bytes, $(wc -l < "$f") records (JSONL)"
        done
    else
        dbg "[prep/3] POPE annotations exist"
        for f in "$POPE_DIR"/coco_pope_*.json; do
            [ -f "$f" ] && dbg "[prep/3]  $(basename "$f"): $(wc -c < "$f") bytes" || true
        done
    fi

    if [ -f "$TARBALL" ]; then
        dbg "[prep/tar] Tarball already exists, skipping"
        ls -lh "$TARBALL"
        return
    fi

    dbg "=== [prep] Creating code-only tarball (models + COCO download on Colab) ==="
    TAR_START=$(date +%s)
    tar cf "$TARBALL" \
        --exclude='*/.git' \
        --exclude='*/__pycache__' \
        --exclude='*.pyc' \
        ONLY/ \
        colab.sh \
        data/pope/
    TAR_ELAPSED=$(($(date +%s) - TAR_START))

    dbg "[prep/tar] Tarball created in ${TAR_ELAPSED}s"
    ls -lh "$TARBALL"
    dbg "[prep/tar] Contents:"
    tar tf "$TARBALL" | sed 's/^/  /'
    TOTAL_FILES=$(tar tf "$TARBALL" | wc -l)
    dbg "[prep/tar] ${TOTAL_FILES} files total"
    echo ""
    dbg "=== prep done ==="
}

run() {
    cd "$ROOT"
    dbg "[run] WORKDIR=$(pwd)"
    export PYTHONPATH="${ROOT}/ONLY:${ROOT}/ONLY/eval_bench:${ROOT}/ONLY/experiments:${PYTHONPATH}"
    dbg "[run] PYTHONPATH=${PYTHONPATH}"

    dbg "=== [run/1] Install dependencies (in-place on Colab system Python) ==="
    PIP_START=$(date +%s)
    dbg "[run/1] $(python --version), torch=$(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo 'MISSING')"
    # We fix Colab's system Python 3.12 in place (NOT a venv) so Colab's
    # pre-installed torch + CUDA wheels stay importable. We only swap the few
    # packages that conflict with ONLY's pinned transformers==4.31.0:
    #   - transformers 4.31.0 needs huggingface_hub<1.0 (Colab ships 1.x, which
    #     removed http_get/REGEX_COMMIT_HASH that 4.31.0 imports → ImportError).
    #   - transformers 4.31.0 pins tokenizers<0.14, but 0.13.x has NO cp312
    #     wheel → use 0.19.1 (has cp312 wheel; LLaVA doesn't use the tokenizers
    #     API directly, so the version skew is harmless apart from the check).
    # --no-deps: stop pip's resolver "upgrading" huggingface_hub back to 1.x.
    # --only-binary :all: for tokenizers: never fall back to the Rust sdist.
    pip install -q --no-deps transformers==4.31.0 2>&1 | tail -1
    pip install -q --no-deps huggingface_hub==0.16.4 2>&1 | tail -1
    pip install -q --only-binary :all: --no-deps tokenizers==0.19.1 2>&1 | tail -1
    pip install -q accelerate sentencepiece einops timm peft \
        bitsandbytes scipy opencv-python pycocotools pandas pillow 2>&1 | tail -1
    # hf_transfer: installed but NOT enabled (was causing RuntimeError on some
    # Colab connections). It sits unused unless HF_HUB_ENABLE_HF_TRANSFER is
    # set, which we intentionally don't do. Keep it installed so that toggling
    # back is a no-op.
    pip install -q --only-binary :all: hf_transfer 2>&1 | tail -1
    # Re-pin the trio in case the line above pulled a wrong version transitively.
    pip install -q --no-deps --force-reinstall --only-binary :all: \
        transformers==4.31.0 huggingface_hub==0.16.4 tokenizers==0.19.1 2>&1 | tail -1

    dbg "=== [run/1b] Disable transformers version check (before importing it) ==="
    # transformers 4.31.0's dependency_versions_check.py runs
    # require_version_core(deps["tokenizers"]) at import time and raises
    # ImportError because tokenizers==0.19.1 violates the <0.14 pin. We must
    # neuter that check WITHOUT importing transformers (importing triggers it).
    # find_spec resolves the path WITHOUT executing transformers/__init__.py,
    # so the version check never fires while we patch it.
    DVC=$(python -c "
import importlib.util as u, os
s = u.find_spec('transformers')
print(os.path.join(os.path.dirname(s.origin), 'dependency_versions_check.py'))
" 2>/dev/null)
    if [ -n "$DVC" ] && [ -f "$DVC" ]; then
        cp "$DVC" "${DVC}.bak"
        sed -i 's/require_version_core(deps\[pkg\])/pass  # version check disabled by ONLY setup/' "$DVC"
        dbg "[run/1b] Patched $DVC (orig saved as .bak)"
    else
        dbg "[run/1b] ⚠️  dependency_versions_check.py not found — version check may still fire"
    fi

    dbg "[run/1] versions: $(python -c 'import transformers,huggingface_hub,tokenizers;print(f"tf={transformers.__version__} hub={huggingface_hub.__version__} tok={tokenizers.__version__}")' 2>&1)"
    dbg "[run/1] pip done in $(($(date +%s)-PIP_START))s"

    dbg "=== [run/2] Patch transformers with ONLY modifications ==="
    python -c "
import transformers, shutil
from pathlib import Path
tf = Path(transformers.__file__).parent

# Patch modeling_llama.py
dst = tf / 'models' / 'llama' / 'modeling_llama.py'
src = Path('${ROOT}/ONLY/patches/modeling_llama.py')
old_size = dst.stat().st_size
dst.write_text(src.read_text())
print(f'  Patched modeling_llama.py ({old_size} → {dst.stat().st_size} bytes)')

# (Version check already disabled in [run/1b].) Belt-and-suspenders:
# also blank the whole check loop in case sed missed an alternate form.
(c := tf/'dependency_versions_check.py').write_text(c.read_text().replace('require_version_core(deps[pkg])', 'pass'))
for p in tf.rglob('__pycache__'): shutil.rmtree(p, ignore_errors=True)
print(f'  Version checks disabled, transformers=={transformers.__version__}')
"

    dbg "=== [run/3] Download assets ==="
    # NOTE: we do NOT silence stderr to /dev/null here. Earlier runs died in 1s
    # at the LLaVA download with no visible error because output was swallowed.
    # We keep stdout quiet (no progress-bar spam) but route stderr to a log so
    # real failures surface. hf_transfer is intentionally not used: it's ~2x
    # faster but causes flaky RuntimeError on many Colab connections. The model
    # downloads happen once and persist on the reused VM, so ~20 min is fine.
    DL_ERR="${ROOT}/logs/download_errors.log"
    mkdir -p "${ROOT}/logs"

    # LLaVA
    # A previous failed download (e.g. hf_transfer crash) may have left an empty
    # or stub directory. Check for an actual model file, not just the dir.
    LLAMA_CONFIG="${MODEL_DIR}/config.json"
    if [ -f "$LLAMA_CONFIG" ] && [ "$(stat -c%s "$LLAMA_CONFIG" 2>/dev/null || echo 0)" -gt 100 ]; then
        dbg "[run/3]  ✅ LLaVA exists ($(du -sh "$MODEL_DIR" | cut -f1))"
    else
        if [ -d "$MODEL_DIR" ]; then
            dbg "[run/3]  ⚠️  LLaVA dir exists but empty/stub — re-downloading"
            rm -rf "$MODEL_DIR"
        fi
        dbg "[run/3] Downloading LLaVA-1.5-7b (~14 GB, may take 10-20 min)..."
        mkdir -p "$MODEL_DIR"
        EVAL_START_LLAVA=$(date +%s)
        # Heartbeat: download output is silent (hf_transfer has no progress
        # bars), so print $MODEL_DIR size every 60s to prove it's alive.
        (
            while true; do
                sleep 60
                sz=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
                dbg "[run/3]  (heartbeat) LLaVA download in progress... ${sz:-0}"
            done
        ) &
        HB_PID=$!
        # local_dir_use_symlinks=False: place files DIRECTLY in $MODEL_DIR
        # (default "auto" symlinks big files into the HF cache, so du -sh
        # reports a misleading ~560K while 14 GB sits in the cache).
        #
        # Retry loop: HF API gateway is flaky from some Colab regions (504
        # Gateway Timeout on repo_info). We retry with backoff + optional
        # mirror + longer timeout.
        #   POPE_HF_ENDPOINT=https://hf-mirror.com  — use Chinese mirror
        #   POPE_HF_TIMEOUT=120                     — increase API timeout
        DL_OK=0
        for ATTEMPT in 1 2 3; do
            dbg "[run/3]  LLaVA download attempt ${ATTEMPT}/3..."
            if python -c "
import os
endpoint = os.environ.get('POPE_HF_ENDPOINT', '')
if endpoint:
    os.environ['HF_ENDPOINT'] = endpoint
    print(f'  Using HF endpoint: {endpoint}', flush=True)
timeout = os.environ.get('POPE_HF_TIMEOUT', '')
if timeout:
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = timeout
    print(f'  Using HF download timeout: {timeout}s', flush=True)
from huggingface_hub import snapshot_download
snapshot_download('liuhaotian/llava-v1.5-7b', local_dir='$MODEL_DIR',
                  local_dir_use_symlinks=False, resume_download=True)
" >"$DL_ERR" 2>&1; then
                DL_OK=1
                break
            fi
            ELAPSED=$(($(date +%s) - EVAL_START_LLAVA))
            dbg "[run/3]  ❌ Attempt ${ATTEMPT} failed at ${ELAPSED}s — last 5 lines:"
            tail -n 5 "$DL_ERR" | sed 's/^/      /'
            if [ "$ATTEMPT" -lt 3 ]; then
                SLEEP=$((ATTEMPT * 15))
                dbg "[run/3]  Retrying in ${SLEEP}s..."
                sleep "$SLEEP"
            fi
        done
        if [ "$DL_OK" -eq 1 ]; then
            kill "$HB_PID" 2>/dev/null || true
            dbg "[run/3]  ✅ LLaVA done ($(du -sh "$MODEL_DIR" | cut -f1))"
        else
            kill "$HB_PID" 2>/dev/null || true
            dbg "[run/3]  ❌ LLaVA download FAILED after 3 attempts — last 15 lines:"
            tail -n 15 "$DL_ERR" | sed 's/^/      /'
            return 1
        fi
    fi

    # CLIP
    CLIP_CONFIG="${CLIP_DIR}/config.json"
    if [ -f "$CLIP_CONFIG" ] && [ "$(stat -c%s "$CLIP_CONFIG" 2>/dev/null || echo 0)" -gt 100 ]; then
        dbg "[run/3]  ✅ CLIP exists ($(du -sh "$CLIP_DIR" | cut -f1))"
    else
        if [ -d "$CLIP_DIR" ]; then
            dbg "[run/3]  ⚠️  CLIP dir exists but empty/stub — re-downloading"
            rm -rf "$CLIP_DIR"
        fi
        dbg "[run/3] Downloading CLIP (~1.7 GB)..."
        mkdir -p "$CLIP_DIR"
        (
            while true; do
                sleep 60
                sz=$(du -sh "$CLIP_DIR" 2>/dev/null | cut -f1)
                dbg "[run/3]  (heartbeat) CLIP download in progress... ${sz:-0}"
            done
        ) &
        HB_PID=$!
        if python -c "
from huggingface_hub import snapshot_download
snapshot_download('openai/clip-vit-large-patch14-336', local_dir='$CLIP_DIR',
                  local_dir_use_symlinks=False, resume_download=True)
" >"$DL_ERR" 2>&1; then
            kill "$HB_PID" 2>/dev/null || true
            dbg "[run/3]  ✅ CLIP done ($(du -sh "$CLIP_DIR" | cut -f1))"
        else
            kill "$HB_PID" 2>/dev/null || true
            dbg "[run/3]  ❌ CLIP download FAILED — last 15 lines of error log:"
            tail -n 15 "$DL_ERR" | sed 's/^/      /'
            return 1
        fi
    fi

    # COCO val2014
    if [ -d "$COCO_DIR" ]; then
        dbg "[run/3]  ✅ COCO exists ($(ls "$COCO_DIR" | wc -l) images)"
    else
        dbg "[run/3] Downloading COCO val2014 (~4 GB)..."
        mkdir -p "$(dirname "$COCO_DIR")"
        wget http://images.cocodataset.org/zips/val2014.zip -O /tmp/val2014.zip >/dev/null 2>&1
        dbg "[run/3]  Downloaded, checking integrity..."
        unzip -tq /tmp/val2014.zip >/dev/null 2>&1
        unzip -q /tmp/val2014.zip -d "$(dirname "$COCO_DIR")/"
        rm /tmp/val2014.zip
        dbg "[run/3]  ✅ COCO done ($(ls "$COCO_DIR" | wc -l) images)"
    fi

    # POPE annotations.
    # Format check: a reused VM may hold a POPE file written by an OLD version
    # of this script (single-line JSON ARRAY + int labels). That format breaks
    # pope_loader.py (line['image'] on a list, and label normalization). So we
    # regenerate whenever the file is MISSING or not valid JSON Lines (i.e. the
    # first line doesn't start with '{').
    POPE_FILE1="${POPE_DIR}/coco_pope_adversarial.json"
    POPE_OK=0
    if [ -f "$POPE_FILE1" ]; then
        # 2>/dev/null + awk so a corrupt/empty file can't crash set -e.
        if [ "$(awk 'NR==1{print substr($0,1,1); exit}' "$POPE_FILE1" 2>/dev/null)" = "{" ]; then
            POPE_OK=1
        fi
    fi
    if [ "$POPE_OK" -eq 1 ]; then
        dbg "[run/3]  ✅ POPE exists ($(wc -l < "$POPE_FILE1") adversarial records, JSONL)"
    else
        [ -f "$POPE_FILE1" ] && dbg "[run/3]  ⚠️  Existing POPE file is old array format — regenerating as JSONL..." || dbg "[run/3] Downloading POPE annotations..."
        rm -rf "$POPE_DIR"
        mkdir -p "$POPE_DIR"
        pip install -q pandas pyarrow 2>&1 | tail -1
        python -c "
import pandas as pd, json
from huggingface_hub import hf_hub_download
for tp, pp in [('random','data/test-00000-of-00003.parquet'), ('popular','data/test-00001-of-00003.parquet'), ('adversarial','data/test-00002-of-00003.parquet')]:
    print(f'  {tp}...', flush=True)
    lp = hf_hub_download(repo_id='lmms-lab/POPE', filename=pp, repo_type='dataset')
    df = pd.read_parquet(lp)
    # JSON Lines (one dict per line) — pope_loader.py reads line-by-line.
    # label = answer string ('yes'/'no'); the loader converts 'no'→0 else→1.
    records = [{'question_id': int(r.question_id), 'image': str(r.image_source) + '.jpg', 'text': str(r.question), 'answer': str(r.answer), 'label': str(r.answer)} for _, r in df.iterrows()]
    with open('$POPE_DIR/coco_pope_' + tp + '.json', 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
    print(f'  {tp}: {len(records)} records (JSONL)', flush=True)
"
        dbg "[run/3]  ✅ POPE done ($(wc -l < "${POPE_DIR}/coco_pope_adversarial.json") adversarial records)"
    fi

    dbg "=== [run/4] Run POPE evaluation ==="
    # Run all three POPE setups (random, popular, adversarial). Each uses its own
    # JSON file + --type (the type only names the per-run log subdirectory inside
    # logs/pope/<model>/ONLY_coco_<type>_...; the questions come from --pope_path)
    # and its own streamed tee file (proposal${POPE_PROPOSAL}_result_<type>.txt), so the three
    # runs never clobber each other.
    #
    # The model is reloaded per setup (~1-2 min each). The alternative — one
    # process iterating over three datasets — would need a refactor of the eval
    # core (which hardcodes a single pope_path), so we loop in bash instead.
    #
    # max_new_tokens=8: KEEP AT 8. An earlier attempt to cut this to 3 (to save
    # decode time) tanked accuracy because the eval uses STOCHASTIC sampling
    # (do_sample=True, temperature=1.0, top_p=1). recorder() scores the answer
    # as "no" ONLY if a negation word (No/not/no/NO, "...n't") appears in the
    # generated text, else defaults to "yes". With only 3 tokens at temp=1 the
    # negation token frequently lands AFTER the cutoff → the answer is recorded
    # as "yes" even when the model meant "no". On the random split this produced
    # 586 false-yes (vs 252 at 8 tokens) and acc collapsed 87->76. The ~2.6x
    # decode speedup is NOT worth it. If you want speed, disable sampling
    # (temperature=0 / do_sample=False) instead — but that changes the protocol
    # vs the 8-token adversarial baseline, so the results wouldn't be comparable.
    #
    # Override the setup list, e.g. to skip re-running adversarial you already
    # have:  POPE_SETUPS="random popular" bash colab.sh run
    # For quick iteration (300/3000 questions): POPE_SHORT=1 bash colab.sh run
    POPE_SETUPS="${POPE_SETUPS:-random popular adversarial}"
    POPE_SHORT="${POPE_SHORT:-0}"
    POPE_TOKENS="${POPE_TOKENS:-8}"
    POPE_ALPHA="${POPE_ALPHA:-0.2}"    # <0 = adaptive per-step alpha (option 2)
    POPE_DEBUG_TVD="${POPE_DEBUG_TVD:-0}"
    POPE_PROPOSAL="${POPE_PROPOSAL:-1}"
    POPE_SCORE_THRESHOLD="${POPE_SCORE_THRESHOLD:-0.0}"
    POPE_SCORE_TEMPERATURE="${POPE_SCORE_TEMPERATURE:-1.0}"
    POPE_LAMBDA_DECAY="${POPE_LAMBDA_DECAY:-0.3}"
    POPE_JS_GAMMA="${POPE_JS_GAMMA:-0.6}"
    POPE_MAXQ="${POPE_MAXQ:-0}"       # explicit override; else derived from SHORT below
    mkdir -p "${ROOT}/logs"
    RUN_TS="$(date +%Y-%m-%d_%Hh%Mm%Ss)"
    RUN_DIR="${ROOT}/logs/results_${RUN_TS}"
    mkdir -p "${RUN_DIR}"
    OVERALL_START=$(date +%s)
    OVERALL_EXIT=0

    for SETUP in $POPE_SETUPS; do
        dbg "[run/4] ==========================================="
        POPE_FILE="${POPE_DIR}/coco_pope_${SETUP}.json"
        if [ ! -f "$POPE_FILE" ]; then
            dbg "[run/4] ⚠️  ${POPE_FILE} missing — skipping ${SETUP}"
            continue
        fi
        # POPE files are JSONL (one dict per line), so count lines — NOT json.load,
        # which raises on multi-line JSONL and would print "?".
        POPE_COUNT=$(wc -l < "$POPE_FILE" 2>/dev/null | tr -d ' ')
        dbg "[run/4] >>> Setup: ${SETUP}  (Questions: ${POPE_COUNT}, max_new_tokens=${POPE_TOKENS}, batch_size=1, mask_alpha=${POPE_ALPHA}, proposal=${POPE_PROPOSAL}, js_gamma=${POPE_JS_GAMMA}${POPE_SHORT:+ , SHORT=300})"
        dbg "[run/4] pope_path=${POPE_FILE}"
        dbg "[run/4] log=${RUN_DIR}/proposal${POPE_PROPOSAL}_result_${SETUP}.txt"
        dbg "[run/4] Loading model and starting evaluation..."

        EVAL_START=$(date +%s)
        # Per-setup heartbeat (prints every 60s).
        (
            while true; do
                sleep 60
                elapsed=$(($(date +%s) - EVAL_START))
                dbg "[run/4] (${SETUP}) still running... ${elapsed}s elapsed"
            done
        ) &
        HEARTBEAT_PID=$!

        set +e  # don't abort the loop if one setup fails — capture + continue
        python -u "${ROOT}/ONLY/eval_bench/pope_eval_llava.py" \
            --model_path "${MODEL_DIR}" \
            --model_base "llava" \
            --pope_path "${POPE_FILE}" \
            --data_path "${COCO_DIR}" \
            --log_path "${ROOT}/logs" \
            --use_only "True" \
            --type "${SETUP}" \
            --dataset_name "coco" \
            --enhance_layer_index "0" \
            --mask_alpha "${POPE_ALPHA}" \
            --temperature "1.0" \
            --top_p "1" \
            --max_new_tokens "${POPE_TOKENS}" \
            --max_questions "$( if [ "$POPE_MAXQ" -gt 0 ] 2>/dev/null; then echo "$POPE_MAXQ"; elif [ "$POPE_SHORT" = "1" ]; then echo "300"; else echo "0"; fi )" \
            --debug_tvd "$([ "$POPE_DEBUG_TVD" = "1" ] && echo "True" || echo "False")" \
            --proposal "${POPE_PROPOSAL}" \
            --score_threshold "${POPE_SCORE_THRESHOLD}" \
            --score_temperature "${POPE_SCORE_TEMPERATURE}" \
            --lambda_decay "${POPE_LAMBDA_DECAY}" \
            --js_gamma "${POPE_JS_GAMMA}" \
            --batch_size "1" \
            --num_workers "1" \
            --seed "42" \
            2>&1 | tee "${RUN_DIR}/proposal${POPE_PROPOSAL}_result_${SETUP}.txt"
        EVAL_EXIT=${PIPESTATUS[0]}
        set -e

        kill "$HEARTBEAT_PID" 2>/dev/null || true
        EVAL_ELAPSED=$(($(date +%s) - EVAL_START))
        dbg "[run/4] ${SETUP}: Python exit code ${EVAL_EXIT}, elapsed ${EVAL_ELAPSED}s"
        if [ "$EVAL_EXIT" -ne 0 ]; then
            OVERALL_EXIT=$EVAL_EXIT
            dbg "[run/4] ⚠️  ${SETUP} failed — continuing to next setup"
        fi
    done

    # Logs packaging — always runs, even on eval failure. Tars the whole logs/
    # dir so ALL per-type result files + the structured logs/pope/<model>/
    # ONLY_coco_<type>_... subdirs are captured.
    dbg "=== [run/5] Packaging logs ==="
    cd "${ROOT}/logs"
    tar czf "/content/results.tar.gz" . 2>/dev/null || true
    cd "${ROOT}"
    dbg "[run/5] Created /content/results.tar.gz ($(stat -c%s /content/results.tar.gz 2>/dev/null || echo 0) bytes)"

    OVERALL_ELAPSED=$(($(date +%s) - OVERALL_START))
    dbg "=== [run/done] All POPE setups finished in ${OVERALL_ELAPSED}s (overall exit ${OVERALL_EXIT}) ==="
    echo ""
    echo "Results (per setup) — ${RUN_DIR}:"
    for SETUP in $POPE_SETUPS; do
        rf="${RUN_DIR}/proposal${POPE_PROPOSAL}_result_${SETUP}.txt"
        if [ -f "$rf" ]; then
            echo "  ${SETUP}: ${rf}"
            # Final accuracy = the LAST 'acc:' line (the per-question prints show a
            # running acc; the final summary line has the overall number).
            grep "acc:" "$rf" 2>/dev/null | tail -1 | sed 's/^/        /' || true
        else
            echo "  ${SETUP}: (no output file)"
        fi
    done
    echo "---"
    set -e

    return ${OVERALL_EXIT}
}

finish() {
    local session="$1"
    if [ -z "$session" ]; then
        echo "Usage: bash colab.sh finish <session-name>"
        exit 1
    fi
    dbg "=== Downloading results from session: ${session} ==="
    mkdir -p "${ROOT}/logs"
    colab download -s "$session" /content/results.tar.gz "${ROOT}/logs/results.tar.gz" || dbg "⚠️  Download failed"
    if [ -f "${ROOT}/logs/results.tar.gz" ]; then
        tar xzf "${ROOT}/logs/results.tar.gz" -C "${ROOT}/logs/"
        dbg "Downloaded and extracted to ${ROOT}/logs/"
    fi
    dbg "=== Stopping session: ${session} ==="
    colab stop -s "$session" || dbg "⚠️  Stop failed"
    dbg "=== Done ==="
}

case "${1:-}" in
    prep) prep ;;
    run)  run  ;;
    finish) finish "$2" ;;
    *)
        echo "Usage: bash colab.sh prep|run|finish"
        echo ""
        echo "  prep   — local: download assets + create colab_bundle.tar.gz"
        echo "  run    — Colab: pip install + patch transformers + evaluate"
        echo "  finish — local: download logs + stop session for a given session name"
        exit 1
        ;;
esac
