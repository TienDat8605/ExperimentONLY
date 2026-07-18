#!/bin/bash
# One-shot: prep locally → upload to Colab → run eval → download results.
#
# SESSION REUSE: instead of creating a fresh VM every run (which re-downloads
# the 14 GB LLaVA model + 4 GB COCO + deps), this keeps a stable-named T4
# session alive across runs. On a reused session it only:
#   1. restarts the kernel (clears Python/GPU state — cheap),
#   2. re-uploads the tiny code-only tarball (overwrites ONLY/ + colab.sh),
#   3. re-extracts and re-runs. colab.sh run's if [ -d ... ] guards skip
#      deps/models/data that already persist on the VM, so a repeated run is
#      near-instant up to model load.
# The session is NOT stopped at the end (kept for next run). Use --stop to stop
# it, or --fresh to force a brand-new VM (deletes + recreates).
#
# Usage:
#   bash run_all.sh            # reuse/create the stable session + run all 3 setups
#   bash run_all.sh --stop     # stop the stable session after this run
#   bash run_all.sh --fresh    # delete any existing session, create a fresh VM
#   bash run_all.sh --setups random popular   # run/verify only these POPE setups

#   bash run_all.sh --benchmarks pope chair mme_hallucination   # run specific benchmarks
#   bash run_all.sh --benchmarks pope   # run only POPE
#
# Requires: colab CLI authenticated, free T4 quota available.
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
SESSION="${SESSION:-only-eval}"          # stable name → reused across runs
TARBALL="${ROOT}/colab_bundle.tar.gz"

STOP_AFTER=0
FORCE_FRESH=0
# Default to ALL THREE POPE setups. Override on the CLI:
#   bash run_all.sh --setups random popular
#   bash run_all.sh --setups random popular adversarial
# This is threaded into the Colab runner's env so colab.sh run evaluates exactly
# these setups, AND into download_logs so it verifies exactly these setups.
POPE_SETUPS="random popular adversarial"
POPE_SHORT=0
POPE_TOKENS=8
POPE_ALPHA=0.2
POPE_DEBUG_TVD=0
POPE_MAXQ=0
POPE_PROPOSAL=1
POPE_SCORE_THRESHOLD=0.0
POPE_SCORE_TEMPERATURE=1.0
POPE_LAMBDA_DECAY=0.3
POPE_JS_GAMMA=0.6
BENCHMARKS="pope chair mme_hallucination"

while [ $# -gt 0 ]; do
    case "$1" in
        --stop)  STOP_AFTER=1; shift ;;
        --fresh) FORCE_FRESH=1; shift ;;
        --short) POPE_SHORT=1; shift ;;
        --maxq=*) POPE_MAXQ="${1#--maxq=}"; shift ;;
        --debug) POPE_DEBUG_TVD=1; shift ;;
        --tokens=*) POPE_TOKENS="${1#--tokens=}"; shift ;;
        --alpha=*) POPE_ALPHA="${1#--alpha=}"; shift ;;
        --proposal=*) POPE_PROPOSAL="${1#--proposal=}"; shift ;;
        --score_threshold=*) POPE_SCORE_THRESHOLD="${1#--score_threshold=}"; shift ;;
        --score_temperature=*) POPE_SCORE_TEMPERATURE="${1#--score_temperature=}"; shift ;;
        --lambda_decay=*) POPE_LAMBDA_DECAY="${1#--lambda_decay=}"; shift ;;
        --js_gamma=*) POPE_JS_GAMMA="${1#--js_gamma=}"; shift ;;
        --setups)
            POPE_SETUPS=""; shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                POPE_SETUPS="${POPE_SETUPS} $1"; shift
            done
            POPE_SETUPS="${POPE_SETUPS# }"   # strip leading space
            ;;
        --setups=*)
            POPE_SETUPS="${1#--setups=}"; shift ;;
        --benchmarks)
            BENCHMARKS=""; shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                BENCHMARKS="${BENCHMARKS} $1"; shift
            done
            BENCHMARKS="${BENCHMARKS# }"
            ;;
        --benchmarks=*)
            BENCHMARKS="${1#--benchmarks=}"; shift ;;
        *) echo "Unknown arg: $1" >&2; shift ;;
    esac
done

cd "$ROOT"

dbg() {
    echo "[$(date '+%H:%M:%S')] $*"
}
dbg "POPE_SETUPS = [${POPE_SETUPS}]"
dbg "BENCHMARKS = [${BENCHMARKS}]"

# True if the stable session currently exists + is usable.
create_session() {
    dbg "colab new -s ${SESSION} --gpu T4"
    local output rc attempt
    for attempt in 1 2 3 4 5; do
        output=$(colab new -s "$SESSION" --gpu T4 2>&1) && return 0
        rc=$?
        if echo "$output" | grep -q "TooManyAssignmentsError\|Precondition Failed\|412"; then
            dbg "⚠️  GPU quota full (412). Stopping leftover sessions and retrying..."
            free_gpu_quota
        elif echo "$output" | grep -q "Service Unavailable\|503\|ColabRequestError"; then
            dbg "⚠️  Colab allocator returned 503 (transient capacity). Retry ${attempt}/5 in $((attempt*10))s..."
            sleep $((attempt * 10))
            continue
        else
            echo "$output"
            return $rc
        fi
        dbg "Retry ${attempt}/5: colab new -s ${SESSION} --gpu T4"
        sleep $((attempt * 5))
    done
    dbg "❌  Failed to create session after 5 attempts. Last error:"
    echo "$output" | tail -5
    return 1
}

session_exists() {
    colab sessions 2>&1 | grep -q "\[${SESSION}\]"
}

# Detect a session whose friendly-name binding broke (shows as "[?] <vm-id>"
# in `colab sessions`). The raw VM id is NOT acceptable as a -s target, and
# `colab stop -s <raw-id>` is a no-op, so we can't kill it from here — it GCs
# on Colab's side after a while. We just warn so the user knows a GPU may be
# leaking quota (which can make `colab new` fail with TooManyAssignments).
warn_orphan() {
    local orphans
    orphans=$(colab sessions 2>&1 | grep -F '[?]' | awk '{print $1}')
    if [ -n "$orphans" ]; then
        dbg "⚠️  Orphaned VM(s) with no friendly name present (CLI cannot reach/stop them):"
        echo "$orphans" | sed 's/^/      /'
        dbg "      These GC on Colab's side. If they hold a T4 they may block new allocations (412)."
        dbg "      The script can auto-evict them (see below)."
    fi
}

# Evict orphaned VMs by calling the Colab API directly with the raw endpoint ID,
# bypassing the CLI's name-resolution requirement. This works even when the
# session shows as "[?]" (friendly-name binding lost) — the most common cause
# of 412 GPU-quota-full errors.
evict_orphans() {
    local orphans eps
    # grep -F = fixed string, no regex — avoids BRE/ERE ?-as-quantifier confusion
    orphans=$(colab sessions 2>&1 | grep -F '[?]' | awk '{print $2}')
    if [ -z "$orphans" ]; then
        dbg "  [evict] no orphans found"
        return 0
    fi
    for ep in $orphans; do
        dbg "  [evict] releasing $ep via Colab API..."
        if uv tool run --from google-colab-cli python3 -c "
import sys, os
from colab_cli.auth import get_credentials, AuthProvider
from colab_cli.client import Client, Prod
try:
    creds = get_credentials(os.path.expanduser('~/.colab-cli-oauth-config.json'), AuthProvider.OAUTH2)
    Client(Prod(), creds).unassign('$ep')
    print('OK')
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
" 2>/dev/null; then
            dbg "  [evict] ✅ $ep released"
        else
            dbg "  [evict] ⚠️  failed to release $ep — may need manual web UI"
        fi
        sleep 2
    done
    return 0
}

cleanup_stale_sessions() {
    # Light proactive cleanup: remove old timestamped sessions only.
    set +e
    local stale
    stale=$(colab sessions 2>&1 | grep -oP 'only-eval-\d+' | sort -u)
    if [ -n "$stale" ]; then
        dbg "Cleaning up leftover timestamped session(s):"
        for s in $stale; do
            echo "  → $s"
            colab stop -s "$s" >/dev/null 2>&1
            dbg "     stopped ✓"
        done
        sleep 3
    fi
    set -e
}

# Aggressive — called on 412 GPU-quota-full. Stop ALL visible sessions
# (friendly-named via CLI, plus orphans via API) so the next `colab new`
# has quota.
free_gpu_quota() {
    local target="${1:-$SESSION}"
    set +e

    # ---- Layer 1: stop every named session via the CLI -------------------
    local named
    named=$(colab sessions 2>&1 | grep -oP '\[\K[^]]+' | grep -vFx "$target" | sort -u)
    if [ -n "$named" ]; then
        dbg "Stopping leftover session(s) blocking T4 quota:"
        while IFS= read -r s; do
            [ -z "$s" ] && continue
            echo "  → $s"
            colab stop -s "$s" >/dev/null 2>&1
            sleep 1
        done <<< "$named"
    else
        dbg "  [named] no other named sessions to stop (besides '${target}')"
    fi

    # ---- Layer 2: evict orphan (no-friendly-name) sessions via API -------
    evict_orphans

    sleep 3
    set -e
}

# Reuse the existing session, or create one if none / if --fresh.
ensure_session() {
    if [ "$FORCE_FRESH" -eq 1 ]; then
        dbg "--fresh: stopping any existing session '${SESSION}'..."
        colab stop -s "$SESSION" >/dev/null 2>&1 || true
        sleep 3
        create_session
        dbg "Waiting 30s for fresh VM to init..."
        sleep 30
        return
    fi
    if session_exists; then
        dbg "Reusing existing session '${SESSION}' (no VM recreation)."
    else
        dbg "No existing session '${SESSION}'; creating a fresh T4 VM..."
        cleanup_stale_sessions
        create_session
        dbg "Waiting 30s for fresh VM to init..."
        sleep 30
    fi
    colab status -s "$SESSION" || true
}

# Runner script executed on Colab via `colab exec`. It re-extracts the uploaded
# tarball (overwriting ONLY/ + colab.sh) and re-runs the eval. Kernel restart
# is done BEFORE this runs (by run_all.sh) so Python state is already clean.
write_colab_runner() {
    cat > /tmp/colab_run.py << 'PYEOF'
"""Colab runner: extract uploaded tarball, run colab.sh run."""
import subprocess, os, sys, time, threading

# Injected from run_all.sh (the local --setups list). This forwards the exact
# setups the user requested into the VM's colab.sh run. Placeholder replaced
# by sed below.
POPE_SETUPS_OUTER = "__POPE_SETUPS_PLACEHOLDER__"
POPE_SHORT_OUTER = "__POPE_SHORT_PLACEHOLDER__"
POPE_TOKENS_OUTER = "__POPE_TOKENS_PLACEHOLDER__"
POPE_ALPHA_OUTER = "__POPE_ALPHA_PLACEHOLDER__"
POPE_DEBUG_TVD_OUTER = "__POPE_DEBUG_TVD_PLACEHOLDER__"
POPE_MAXQ_OUTER = "__POPE_MAXQ_PLACEHOLDER__"
POPE_PROPOSAL_OUTER = "__POPE_PROPOSAL_PLACEHOLDER__"
POPE_SCORE_THRESHOLD_OUTER = "__POPE_SCORE_THRESHOLD_PLACEHOLDER__"
POPE_SCORE_TEMPERATURE_OUTER = "__POPE_SCORE_TEMPERATURE_PLACEHOLDER__"
POPE_LAMBDA_DECAY_OUTER = "__POPE_LAMBDA_DECAY_PLACEHOLDER__"
POPE_JS_GAMMA_OUTER = "__POPE_JS_GAMMA_PLACEHOLDER__"
BENCHMARKS_OUTER = "__BENCHMARKS_PLACEHOLDER__"

os.chdir("/content")

def dbg(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def keepalive():
    """Print a heartbeat every 60s so Colab doesn't idle-timeout the session."""
    while not done.is_set():
        time.sleep(60)
        dbg("keepalive -- still running inside colab.sh run")
    dbg("keepalive -- done")

done = threading.Event()
threading.Thread(target=keepalive, daemon=True).start()

dbg("=== colab_run.py started ===")
dbg(f"Files: {[f for f in os.listdir('/content') if not f.startswith('.')]}")

dbg("[1/3] Extracting uploaded colab_bundle.tar.gz (overwrites ONLY/ + colab.sh)...")
# Overwrite on re-extraction so code updates land on a reused session.
subprocess.run(["tar", "xf", "colab_bundle.tar.gz"], check=True)
dbg(f"Extracted: {[f for f in os.listdir('/content') if not f.startswith('.')]}")

dbg("[2/3] Running: bash colab.sh run")
dbg("--- stdout/stderr from colab.sh run (colab.sh already timestamps) ---")
run_env = dict(os.environ)
# Forward the POPE_SETUPS list selected locally into the VM subprocess env,
# so colab.sh run evaluates exactly the setups the user asked for.
run_env["POPE_SETUPS"] = POPE_SETUPS_OUTER
run_env["POPE_SHORT"] = str(POPE_SHORT_OUTER)
run_env["POPE_TOKENS"] = str(POPE_TOKENS_OUTER)
run_env["POPE_ALPHA"] = str(POPE_ALPHA_OUTER)
run_env["POPE_DEBUG_TVD"] = str(POPE_DEBUG_TVD_OUTER)
run_env["POPE_MAXQ"] = str(POPE_MAXQ_OUTER)
run_env["POPE_PROPOSAL"] = str(POPE_PROPOSAL_OUTER)
run_env["POPE_SCORE_THRESHOLD"] = str(POPE_SCORE_THRESHOLD_OUTER)
run_env["POPE_SCORE_TEMPERATURE"] = str(POPE_SCORE_TEMPERATURE_OUTER)
run_env["POPE_LAMBDA_DECAY"] = str(POPE_LAMBDA_DECAY_OUTER)
run_env["POPE_JS_GAMMA"] = str(POPE_JS_GAMMA_OUTER)
	run_env["BENCHMARKS"] = BENCHMARKS_OUTER
dbg(f"Forwarding POPE_SETUPS={run_env['POPE_SETUPS']!r} into colab.sh run")
dbg(f"Forwarding POPE_SHORT={run_env['POPE_SHORT']!r} into colab.sh run")
dbg(f"Forwarding POPE_TOKENS={run_env['POPE_TOKENS']!r} into colab.sh run")
dbg(f"Forwarding POPE_ALPHA={run_env['POPE_ALPHA']!r} into colab.sh run")
dbg(f"Forwarding POPE_DEBUG_TVD={run_env['POPE_DEBUG_TVD']!r} into colab.sh run")
	dbg(f"Forwarding BENCHMARKS={run_env[\"BENCHMARKS\"]!r} into colab.sh run")
proc = subprocess.Popen(
    ["bash", "colab.sh", "run"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    bufsize=1, text=True,
    env=run_env,
)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
done.set()
dbg("--- colab.sh run finished ---")
dbg(f"colab.sh run exit code: {proc.returncode}")
sys.exit(proc.returncode)
PYEOF
    # Substitute the local --setups list into the runner Python. POPE_SETUPS is
    # embedded as a Python string literal, so escape any embedded quotes
    # (setups are simple words, so this is defensive — but correct regardless).
    esc=$(printf '%s' "$POPE_SETUPS" | sed 's/\\/\\\\/g; s/"/\\"/g')
    sed -i "s/__POPE_SETUPS_PLACEHOLDER__/$esc/" /tmp/colab_run.py
    sed -i "s/__POPE_SHORT_PLACEHOLDER__/$POPE_SHORT/" /tmp/colab_run.py
    sed -i "s/__POPE_TOKENS_PLACEHOLDER__/$POPE_TOKENS/" /tmp/colab_run.py
    sed -i "s/__POPE_ALPHA_PLACEHOLDER__/$POPE_ALPHA/" /tmp/colab_run.py
    sed -i "s/__POPE_DEBUG_TVD_PLACEHOLDER__/$POPE_DEBUG_TVD/" /tmp/colab_run.py
    sed -i "s/__POPE_MAXQ_PLACEHOLDER__/$POPE_MAXQ/" /tmp/colab_run.py
    sed -i "s/__POPE_PROPOSAL_PLACEHOLDER__/$POPE_PROPOSAL/" /tmp/colab_run.py
    sed -i "s/__POPE_SCORE_THRESHOLD_PLACEHOLDER__/$POPE_SCORE_THRESHOLD/" /tmp/colab_run.py
    sed -i "s/__POPE_SCORE_TEMPERATURE_PLACEHOLDER__/$POPE_SCORE_TEMPERATURE/" /tmp/colab_run.py
    sed -i "s/__POPE_LAMBDA_DECAY_PLACEHOLDER__/$POPE_LAMBDA_DECAY/" /tmp/colab_run.py
    sed -i "s/__POPE_JS_GAMMA_PLACEHOLDER__/$POPE_JS_GAMMA/" /tmp/colab_run.py
    sed -i "s/__BENCHMARKS_PLACEHOLDER__/$BENCHMARKS/" /tmp/colab_run.py
}

# Download and VERIFY logs from a session. This is hardened against the
# failure mode that bit us on the adversarial run: the results.tar.gz
# download silently returned nothing (or a STALE tarball from a prior run)
# and the friendly session name was lost — yet the full per-question log sat
# safely in the local CLI history the whole time.
#
# Strategy (each layer is tried in order; first one that yields VERIFIED
# per-setup final-acc lines wins):
#   1. results.tar.gz  — retry the download up to 5x with waits, require a
#                        non-empty valid gzip, extract, then VERIFY every
#                        requested setup's 'acc:' summary line is present
#                        in its proposal1_result_<type>.txt. A stale or
#                        incomplete tarball is rejected.
#   2. per-file        — download each proposal1_result_<type>.txt
#                        individually and verify the acc line.
#   3. local history   — recover from ~/.config/colab-cli/history/ via
#                        recover_pope_log.py. Works even if the VM/session
#                        name is gone, because `colab exec` logged the
#                        streamed output locally.
#
# POPE_SETUPS: space-separated list to verify (defaults to all 3). Export it
# before calling, e.g. exported by the Colab runner; else defaults here.
download_logs() {
    local session="$1"
    local setups="${POPE_SETUPS:-random popular adversarial}"
    mkdir -p "${ROOT}/logs"
    local TB="${ROOT}/logs/results.tar.gz"

    # ---- helper: did a setup COMPLETE? ------------------------------------
    # The eval prints a per-question RUNNING 'acc:' line ~3000 times (one per
    # question), so a bare 'acc:' match would pass even on a TRUNCATED run.
    # The real completion marker is the timestamped FINAL summary that
    # logger.info emits once at the very end:  [YYYY-MM-DD HH:MM:SS] acc: ...
    # Require that line — it proves the run finished, not just started.
    has_acc() { [ -f "$1" ] && grep -qE '^\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]+\] acc:' "$1" 2>/dev/null; }

    # ---- helper: count how many requested setups are verified present ----
    verify_all() {
        local found=0
        for s in $setups; do
            has_acc "${ROOT}/logs/proposal${POPE_PROPOSAL}_result_${s}.txt" && found=$((found+1))
        done
        echo "$found"
    }

    # NOTE on the `colab` CLI: on this installation `colab download`/`colab ls`
    # cannot see NOTEBOOK RUNTIME files (only the uploaded-file store), so
    # runtime-written artifacts like /content/results.tar.gz and
    # /content/logs/proposal1_result_*.txt return "File or directory not
    # found" even right after a successful run. Layers 1 & 2 below are kept as
    # best-effort (they'll usually all miss here), and layer 3 — recovery from
    # the local CLI exec-history — is the actual retrieval mechanism that
    # reliably works. We suppress the per-attempt "Download failed" spam and
    # emit a single summary line so the output isn't alarming.
    local download_unavailable=1

    # ---- layer 1: results.tar.gz with retries + gzip validity + verify ---
    rm -f "$TB"
    local attempt
    for attempt in 1 2 3 4 5; do
        if colab download -s "$session" /content/results.tar.gz "$TB" >/dev/null 2>&1; then
            download_unavailable=0
            # Non-empty AND valid gzip (gzip -t fails on a stale/partial file).
            if [ -s "$TB" ] && gzip -t "$TB" 2>/dev/null; then
                dbg "  [tarball] got valid gzip ($(stat -c%s "$TB") bytes), extracting"
                # Extract into a temp dir first so a stale tarball can't be
                # half-merged into logs/ and falsely pass verification.
                local tmpx; tmpx="$(mktemp -d)"
                if tar xzf "$TB" -C "$tmpx" 2>/dev/null; then
                    # Move per-setup result files (and any pope/ structured tree)
                    # into logs/. Files are now inside dated results_<ts>/ subdirs,
                    # so look in both flat and nested locations.
                    cp -f "$tmpx"/proposal${POPE_PROPOSAL}_result_*.txt "${ROOT}/logs/" 2>/dev/null || true
                    cp -f "$tmpx"/results_*/proposal${POPE_PROPOSAL}_result_*.txt "${ROOT}/logs/" 2>/dev/null || true
                    cp -rf "$tmpx"/pope "${ROOT}/logs/" 2>/dev/null || true
                    cp -rf "$tmpx"/results_* "${ROOT}/logs/" 2>/dev/null || true
                    rm -rf "$tmpx"
                    if [ "$(verify_all)" -eq "$(echo $setups | wc -w)" ]; then
                        dbg "  [tarball] ✅ verified all setups present"
                        return 0
                    else
                        dbg "  [tarball] ⚠️  extracted tarball missing some setups' acc lines — trying per-file"
                        break   # tarball not sufficient → fall to layer 2
                    fi
                else
                    dbg "  [tarball] tar extraction failed — retrying"
                    rm -rf "$tmpx"
                fi
            else
                dbg "  [tarball] downloaded file not a valid/complete gzip — retrying"
                rm -f "$TB"
            fi
        fi
        # brief backoff before next attempt (Colab may still be writing it)
        sleep $((attempt * 3))
    done

    # ---- layer 2: per-file download + verify -----------------------------
    # (Quiet: if tarball layer surfaced nothing, per-file will also miss on a
    # runtime-blind CLI. Suppress per-attempt noise; print only the outcome.)
    local got=0
    for s in $setups; do
        local rf="${ROOT}/logs/proposal${POPE_PROPOSAL}_result_${s}.txt"
        [ -f "$rf" ] && has_acc "$rf" && { got=$((got+1)); continue; }
        rm -f "$rf"
        for attempt in 1 2 3; do
            if colab download -s "$session" "/content/logs/proposal${POPE_PROPOSAL}_result_${s}.txt" "$rf" >/dev/null 2>&1 && has_acc "$rf"; then
                download_unavailable=0
                got=$((got+1)); break
            fi
            sleep 2
        done
    done
    if [ "$got" -eq "$(echo $setups | wc -w)" ]; then
        dbg "  [per-file] ✅ downloaded + verified all $got setup(s)"
        return 0
    elif [ "$got" -gt 0 ]; then
        dbg "  [per-file] partial: $got/$(echo $setups | wc -w) setup(s) verified — history recovery for the rest"
    else
        if [ "$download_unavailable" -eq 1 ]; then
            dbg "  ⚠️  colab file download unavailable on this CLI (runtime files not exposed) — using history recovery"
        else
            dbg "  [per-file] no setups verified — using history recovery for all"
        fi
    fi

    # ---- layer 3: local CLI-history recovery (the last-time safety net) -
    dbg "  [history] recovering from ~/.config/colab-cli/history/ via recover_pope_log.py"
    if [ -f "${ROOT}/recover_pope_log.py" ]; then
        if python3 "${ROOT}/recover_pope_log.py" 2>&1 | sed 's/^/    /'; then
            # Copy recovered <setup>_recovered.txt -> proposal${POPE_PROPOSAL}_result_<setup>.txt
            # so downstream tooling finds one canonical file per setup.
            local recov=0
            for s in $setups; do
                local rc="${ROOT}/logs/pope_${s}_recovered.txt"
                if [ -f "$rc" ] && has_acc "${rc}"; then
                    # Only overwrite if the live download is missing the acc line.
                    if ! has_acc "${ROOT}/logs/proposal${POPE_PROPOSAL}_result_${s}.txt"; then
                        cp -f "$rc" "${ROOT}/logs/proposal${POPE_PROPOSAL}_result_${s}.txt"
                    fi
                    recov=$((recov+1))
                fi
            done
            if [ "$recov" -gt 0 ]; then
                dbg "  [history] ✅ recovered $recov setup(s) from local CLI history"
                return 0
            fi
        fi
    else
        dbg "  [history] ⚠️  recover_pope_log.py not found — cannot recover from history"
    fi

    dbg "  ⚠️  Download verification incomplete; checkpointing what we have"
    return 0   # not fatal — run_all.sh stays usable; the eval stdout above has the live results
}

# On ANY error: capture exit code, try to download logs, but DO NOT stop the
# reusable session (so a quick fix + re-run stays fast). Only --stop/--fresh
# tear the session down.
cleanup_on_error() {
    local ec=$?
    dbg "=== ⚠️  run_all.sh failed (exit $ec) — downloading any available logs ==="
    if colab status -s "$SESSION" >/dev/null 2>&1; then
        download_logs "$SESSION"
        dbg "Session '${SESSION}' left running for fast re-run (use --stop to stop it)."
    fi
    exit "$ec"
}
trap cleanup_on_error ERR

dbg "=== run_all.sh starting ==="
echo "Session: ${SESSION}${FORCE_FRESH:+ (FRESH)}${STOP_AFTER:+ (will stop after)}${POPE_SHORT:+ (SHORT=300)} tokens=${POPE_TOKENS} alpha=${POPE_ALPHA} proposal=${POPE_PROPOSAL}${POPE_DEBUG_TVD:+ debug=1}${POPE_MAXQ:+ maxq=${POPE_MAXQ}}"
echo ""

if [ -f "$TARBALL" ]; then
    # Re-pack if any source file is newer than the tarball.
    newest_src=$(find ONLY/ colab.sh -not -path '*__pycache__*' -not -name '*.pyc' -newer "$TARBALL" -type f 2>/dev/null | head -1)
    if [ -n "$newest_src" ]; then
        dbg "=== [1/3] Source changed ($newest_src newer than tarball), re-packing ==="
        rm -f "$TARBALL"
        bash colab.sh prep
    else
        dbg "=== [1/3] Tarball exists and up-to-date, skipping prep ==="
        ls -lh "$TARBALL"
    fi
else
    dbg "=== [1/3] Prep: download + create tarball ==="
    bash colab.sh prep
fi

dbg "=== [2/3] Ensure Colab T4 session ==="
ensure_session

# Upload the code-only tarball as a SINGLE file (no 100KB chunking — the tarball
# is tiny and `colab upload` handles it fine). Using a fixed remote name means a
# reused session just overwrites the old copy.
#
# On a long-lived session the friendly-name binding can silently break (the
# session goes "[?]" in `colab sessions`, or `colab upload` returns "File or
# directory not found" / 404). That is NOT a path error — it means the CLI can
# no longer reach the session. The only cure is a fresh VM. So if the first
# upload fails, we force a brand-new session and retry once before giving up.
# This avoids the recurring "upload failed → whole run aborts" loop.
upload_tarball() {
    # Returns 0 on success. Captures stderr (the "File or directory not found"
    # / "Upload failed" messages) so the caller can log the reason on failure.
    colab upload -s "$SESSION" "$TARBALL" "/content/colab_bundle.tar.gz" 2>&1
}
dbg "[2/3] Uploading code tarball ($(stat -c%s "$TARBALL") bytes)..."
if ! upload_out=$(upload_tarball); then
    dbg "[2/3] ⚠️  upload failed — session binding likely lost (orphaned/dead)."
    dbg "      ($upload_out)"
    dbg "[2/3] Force-creating a fresh session and retrying upload..."
    warn_orphan
    # Drop the dead binding if any, then make a brand-new VM.
    colab stop -s "$SESSION" >/dev/null 2>&1 || true
    sleep 3
    create_session || { dbg "❌  could not create a fresh session; aborting"; exit 1; }
    dbg "Waiting 30s for fresh VM to init..."
    sleep 30
    if ! upload_out=$(upload_tarball); then
        dbg "❌  upload still failing on fresh session:"
        echo "$upload_out" | sed 's/^/      /'
        exit 1
    fi
fi
dbg "[2/3] Upload done"

# Restart the kernel so the run starts from clean Python/GPU state WITHOUT
# recreating the VM (keeps installed packages + downloaded models on disk).
# On a fresh session this is a harmless no-op-ish reset.
dbg "=== [3/3] Restart kernel (clean Python/GPU state, keep VM) ==="
colab restart-kernel -s "$SESSION" || dbg "⚠️  restart-kernel failed (continuing on current kernel)"
sleep 5

dbg "=== [3/3] Run evaluation on Colab ==="
write_colab_runner
EXIT_CODE=0
colab exec -s "$SESSION" --file /tmp/colab_run.py --timeout 7200 || EXIT_CODE=$?

echo "------------------------------------------------------------"
dbg "=== Downloading results ==="
download_logs "$SESSION"

if [ "$STOP_AFTER" -eq 1 ]; then
    dbg "=== --stop: stopping session ${SESSION} ==="
    colab stop -s "$SESSION" || dbg "⚠️  Stop failed"
    dbg "=== ✅ run_all.sh done (session stopped) ==="
elif [ "$EXIT_CODE" -eq 0 ]; then
    dbg "=== ✅ run_all.sh done (session '${SESSION}' kept for fast re-run) ==="
else
    dbg "=== ⚠️  colab exec exited with code ${EXIT_CODE} (session kept — fix & re-run) ==="
fi
exit $EXIT_CODE
