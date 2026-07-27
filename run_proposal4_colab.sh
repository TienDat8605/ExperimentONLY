#!/bin/bash
# ==============================================================================
# Run Proposal 4 (Static/Adaptive 3-Branch Dual-Masking) on Google Colab
# ==============================================================================
#
# Usage:
#   bash run_proposal4_colab.sh --proposal=4    # Proposal 4a (Static 3-branch)
#   bash run_proposal4_colab.sh --proposal=5    # Proposal 4b (Adaptive 3-branch)
#
# Optional flags:
#   --short                 Evaluate on 300 questions for quick testing
#   --setups "adversarial"  Specific POPE setup (default: "adversarial")
#   --session "only-eval"   Colab session name (default: "only-eval")
#   --stop                  Stop Colab session after run completes
#
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
SESSION="${SESSION:-only-eval}"
TARBALL="${ROOT}/colab_bundle.tar.gz"

PROPOSAL=4
SHORT=1
SETUPS="random popular adversarial"
BENCHMARKS="pope"
STOP_AFTER=0
DEBUG_TVD=1

ALPHA_1=""
ALPHA_2=""
ALPHA_3=""
ALPHA_4=""
GAMMA_1=""
GAMMA_2=""

while [ $# -gt 0 ]; do
    case "$1" in
        --proposal=*)   PROPOSAL="${1#--proposal=}"; shift ;;
        --proposal)     PROPOSAL="$2"; shift 2 ;;
        --setups=*)     SETUPS="${1#--setups=}"; shift ;;
        --setups)       SETUPS="$2"; shift 2 ;;
        --benchmarks=*) BENCHMARKS="${1#--benchmarks=}"; shift ;;
        --benchmarks)   BENCHMARKS="$2"; shift 2 ;;
        --session=*)    SESSION="${1#--session=}"; shift ;;
        --session)      SESSION="$2"; shift 2 ;;
        --alpha_1=*)    ALPHA_1="${1#--alpha_1=}"; shift ;;
        --alpha_1)      ALPHA_1="$2"; shift 2 ;;
        --alpha_2=*)    ALPHA_2="${1#--alpha_2=}"; shift ;;
        --alpha_2)      ALPHA_2="$2"; shift 2 ;;
        --alpha_3=*)    ALPHA_3="${1#--alpha_3=}"; shift ;;
        --alpha_3)      ALPHA_3="$2"; shift 2 ;;
        --alpha_4=*)    ALPHA_4="${1#--alpha_4=}"; shift ;;
        --alpha_4)      ALPHA_4="$2"; shift 2 ;;
        --gamma_1=*)    GAMMA_1="${1#--gamma_1=}"; shift ;;
        --gamma_1)      GAMMA_1="$2"; shift 2 ;;
        --gamma_2=*)    GAMMA_2="${1#--gamma_2=}"; shift ;;
        --gamma_2)      GAMMA_2="$2"; shift 2 ;;
        --short)        SHORT=1; shift ;;
        --full)         SHORT=0; shift ;;
        --stop)         STOP_AFTER=1; shift ;;
        *) echo "Unknown arg: $1" >&2; shift ;;
    esac
done

cd "$ROOT"

dbg() {
    echo "[$(date '+%H:%M:%S')] $*"
}

dbg "=== [1/3] Creating lightweight code bundle (models/data download on Colab) ==="
rm -f "$TARBALL"
mkdir -p "${ROOT}/data/coco" "${ROOT}/data/pope" "${ROOT}/data/chair" "${ROOT}/data/mme_hallucination"

tar cf "$TARBALL" \
    --exclude='*/.git' \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    ONLY/ \
    colab.sh \
    data/

dbg "Code bundle created: $(ls -lh "$TARBALL" | awk '{print $5}')"

ensure_session() {
    local sess="$1"
    for attempt in 1 2 3 4 5; do
        dbg "Attempting to create T4 GPU session '${sess}' (attempt ${attempt}/5)..."
        if colab new -s "$sess" --gpu T4; then
            dbg "Session '${sess}' created successfully."
            sleep 15
            return 0
        fi
        dbg "⚠️ Colab service assignment busy/unavailable. Retrying in 15s..."
        sleep 15
    done
    return 1
}

dbg "=== [2/3] Checking Colab session '${SESSION}' ==="
if colab sessions 2>&1 | grep -q "\[${SESSION}\]"; then
    dbg "Reusing existing session '${SESSION}'"
else
    ensure_session "$SESSION"
fi

dbg "=== [3/3] Uploading code bundle and running Proposal ${PROPOSAL} ==="
if ! colab upload -s "$SESSION" "$TARBALL" "/content/colab_bundle.tar.gz"; then
    dbg "⚠️ Upload failed (VM disconnected). Re-creating session '${SESSION}'..."
    colab stop -s "$SESSION" 2>/dev/null || true
    ensure_session "$SESSION"
    colab upload -s "$SESSION" "$TARBALL" "/content/colab_bundle.tar.gz"
fi

# Generate isolated python execution script for Colab
cat > /tmp/run_p4.py << PYEOF
import subprocess, os, sys

os.chdir("/content")
print("[Colab] Extracting code bundle...", flush=True)
subprocess.run(["tar", "xf", "colab_bundle.tar.gz"], check=True)

env = dict(os.environ)
env["POPE_PROPOSAL"] = "${PROPOSAL}"
env["POPE_SHORT"] = "${SHORT}"
env["POPE_SETUPS"] = "${SETUPS}"
env["BENCHMARKS"] = "${BENCHMARKS}"
env["POPE_DEBUG_TVD"] = "${DEBUG_TVD}"
env["POPE_ALPHA_1"] = "${ALPHA_1}"
env["POPE_ALPHA_2"] = "${ALPHA_2}"
env["POPE_ALPHA_3"] = "${ALPHA_3}"
env["POPE_ALPHA_4"] = "${ALPHA_4}"
env["POPE_GAMMA_1"] = "${GAMMA_1}"
env["POPE_GAMMA_2"] = "${GAMMA_2}"

print(f"[Colab] Running evaluation for Proposal ${PROPOSAL}...", flush=True)
proc = subprocess.Popen(
    ["bash", "colab.sh", "run"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
    text=True,
    env=env
)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
sys.exit(proc.returncode)
PYEOF

colab exec -s "$SESSION" --file /tmp/run_p4.py --timeout 7200

dbg "=== [4/4] Downloading logs ==="
mkdir -p "${ROOT}/logs"
colab download -s "$SESSION" /content/results.tar.gz "${ROOT}/logs/results.tar.gz" 2>/dev/null || true

if [ ! -f "${ROOT}/logs/results.tar.gz" ] || [ "$(stat -c%s "${ROOT}/logs/results.tar.gz" 2>/dev/null || echo 0)" -lt 100 ]; then
    dbg "Using base64 fallback to recover results.tar.gz from Colab VM..."
    cat > /tmp/b64_dl.py << 'PYEOF'
import base64, os
p = '/content/results.tar.gz'
if os.path.exists(p):
    print("B64_START:" + base64.b64encode(open(p, 'rb').read()).decode())
else:
    print("NO_FILE")
PYEOF
    colab exec -s "$SESSION" --file /tmp/b64_dl.py > /tmp/res.b64 2>/dev/null || true
    if grep -q "B64_START:" /tmp/res.b64; then
        grep "B64_START:" /tmp/res.b64 | sed 's/B64_START://' | base64 -d > "${ROOT}/logs/results.tar.gz"
        dbg "Base64 recovery successful ($(ls -lh "${ROOT}/logs/results.tar.gz" | awk '{print $5}'))"
    fi
fi

if [ -f "${ROOT}/logs/results.tar.gz" ]; then
    tar xf "${ROOT}/logs/results.tar.gz" -C "${ROOT}/logs/" 2>/dev/null || true
    dbg "Extracted log files into ${ROOT}/logs/"
fi

if [ "$STOP_AFTER" -eq 1 ]; then
    colab stop -s "$SESSION" || true
    dbg "Colab session stopped."
fi

dbg "=== Completed Proposal ${PROPOSAL} evaluation ==="
