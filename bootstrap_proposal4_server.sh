#!/usr/bin/env bash
# One-command setup for a fresh/rented CUDA 12.8 GPU server.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${ROOT}/run_proposal4_local.sh"
VENV_DIR="${PROPOSAL4_VENV:-${ROOT}/.venv-proposal4}"

log() {
    printf '[%(%H:%M:%S)T] %s\n' -1 "$*"
}

remove_path_entry() {
    local unwanted="$1" entry filtered=""
    local old_ifs="$IFS"
    IFS=':'
    for entry in $PATH; do
        [[ "$entry" == "$unwanted" ]] && continue
        filtered="${filtered:+${filtered}:}${entry}"
    done
    IFS="$old_ifs"
    PATH="$filtered"
    export PATH
}

log "Loading shell configuration from ~/.bashrc"
if [[ -r "${HOME}/.bashrc" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc"
    set -u
fi

# Load the Conda shell function when an active environment was inherited but
# .bashrc did not initialize Conda for this non-interactive shell.
if [[ -n "${CONDA_PREFIX:-}" ]] && ! declare -F conda >/dev/null && [[ -x "${CONDA_EXE:-}" ]]; then
    set +u
    eval "$("$CONDA_EXE" shell.bash hook)"
    set -u
fi

set +u
while [[ -n "${CONDA_PREFIX:-}" ]] && declare -F conda >/dev/null; do
    log "Deactivating Conda environment ${CONDA_DEFAULT_ENV:-${CONDA_PREFIX}}"
    conda deactivate
done
set -u

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    inherited_venv="$VIRTUAL_ENV"
    log "Deactivating inherited virtual environment ${inherited_venv}"
    if declare -F deactivate >/dev/null; then
        deactivate
    else
        remove_path_entry "${inherited_venv}/bin"
        unset VIRTUAL_ENV
    fi
fi

PYTHON_BIN="${PROPOSAL4_PYTHON:-$(command -v python3 || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
    echo "ERROR: python3 was not found after deactivating the parent environment" >&2
    exit 1
}
export PROPOSAL4_PYTHON="$PYTHON_BIN"
export PROPOSAL4_VENV="$VENV_DIR"

log "Creating the Proposal 4 environment with ${PROPOSAL4_PYTHON}"
bash "$RUNNER" setup

# Activate for all remaining bootstrap steps. The runner also addresses the
# venv explicitly, so later standalone invocations do not require activation.
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log "Activated ${VENV_DIR}"

bash "$RUNNER" download
bash "$RUNNER" check

log "Proposal 4 setup is complete"
printf 'For a new shell, activate it with:\n  source %q\n' "${VENV_DIR}/bin/activate"
printf 'Smoke test:\n  CUDA_VISIBLE_DEVICES=0 bash %q run --setups adversarial --maxq=10\n' "$RUNNER"
