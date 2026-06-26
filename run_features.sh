#!/bin/sh
set -eu

REPO_ROOT="/home/riftuser/gaia_pipeline"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
LLAMA_BUILD_DIR="${REPO_ROOT}/_state/llama.cpp-cuda/build"
LLAMA_SERVER="${LLAMA_BUILD_DIR}/bin/llama-server"
STATE_DIR="${REPO_ROOT}/_state/runner"
export HF_HOME="${REPO_ROOT}/_state/huggingface"
export INSPECT_LOG_DIR="${REPO_ROOT}/_state/inspect-logs"
export PLAYWRIGHT_BROWSERS_PATH="${REPO_ROOT}/playwright-browsers"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export LD_LIBRARY_PATH="${LLAMA_BUILD_DIR}/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export GGML_BACKEND_PATH="${LLAMA_BUILD_DIR}/bin/ggml-cuda-plugin.so"

export BASE_MODEL_API_BASE_URL="http://127.0.0.1:18082/v1"
export BASE_MODEL_API_KEY="local"
export BASE_MODEL_NAME="qwen3.5-9b-gguf-q4"
export HF_TOKEN="$(grep HF_TOKEN "${REPO_ROOT}/.env" | cut -d= -f2 | tr -d '\r')"
export GAIA_TASK="inspect_evals/gaia_level1"
export GAIA_SPLIT="validation"
export GAIA_MAX_CONNECTIONS="1"
export GAIA_MAX_ATTEMPTS="1"
export GAIA_SANDBOX="local"
export SCAFFOLD_PORT="19080"
export SCAFFOLD_API_KEY="local"
export SCAFFOLD_MODEL_NAME="gaia-scaffold-bon"
export SAMPLE_LIMIT="40-53"
export EXPECTED_SAMPLES="14"
export PICSAR_BRANCHES="2"

export GAIA_MODEL_API_KEY="${SCAFFOLD_API_KEY}"
export GAIA_MODEL_BASE_URL="http://127.0.0.1:${SCAFFOLD_PORT}/v1"

FEATURES="
FEATURE_SELF_CONSISTENCY
FEATURE_SHORT_MAK
FEATURE_ACON
FEATURE_ADAPTIVE_BON
FEATURE_BAVT
FEATURE_RANKED_VOTING
FEATURE_PICSAR
FEATURE_MOB
FEATURE_STRUCTURED_NOTES
FEATURE_RESUM
FEATURE_HIAGENT
FEATURE_CONTEXT_COMPACTION
FEATURE_COMPLEXITY_CONSISTENCY
"

RESULTS=""
DONE_DIR="${INSPECT_LOG_DIR}/.done"
mkdir -p "${DONE_DIR}"

check_llama() {
    "${VENV_PYTHON}" - <<'PY'
import os
from urllib.request import Request, urlopen
req = Request("http://127.0.0.1:18082/v1/models")
req.add_header("Authorization", "Bearer local")
try:
    urlopen(req, timeout=5).close()
except Exception:
    raise SystemExit(1)
PY
}

check_scaffold() {
    "${VENV_PYTHON}" - <<'PY'
from urllib.request import urlopen
import os
try:
    urlopen(f"http://127.0.0.1:{os.environ['SCAFFOLD_PORT']}/health", timeout=5).close()
except Exception:
    raise SystemExit(1)
PY
}


# ── Start llama if not running ───────────────────────────────────────────────
if check_llama 2>/dev/null; then
    echo "==> llama-server already running."
else
    echo "==> Starting llama-server..."
    mkdir -p "${STATE_DIR}"
    "${LLAMA_SERVER}" \
        -hf "unsloth/Qwen3.5-9B-GGUF:Q4_K_M" \
        --alias "${BASE_MODEL_NAME}" \
        --host "0.0.0.0" \
        --port "18082" \
        --api-key "${BASE_MODEL_API_KEY}" \
        --ctx-size "131072" \
        --n-gpu-layers "999" \
        --threads "4" \
        --parallel "1" \
        --batch-size "512" \
        --ubatch-size "128" \
        --jinja \
        -fit off \
        > "${STATE_DIR}/llama16GB.stdout" \
        2> "${STATE_DIR}/llama16GB.stderr" &
    LLAMA_PID=$!

    printf "==> Waiting for llama-server (pid %s)..." "${LLAMA_PID}"
    i=0
    while ! check_llama 2>/dev/null; do
        kill -0 "${LLAMA_PID}" 2>/dev/null || { echo " DIED"; tail -5 "${STATE_DIR}/llama16GB.stderr"; exit 1; }
        i=$((i+1))
        [ $i -le 120 ] || { echo " TIMEOUT"; exit 1; }
        printf "."
        sleep 5
    done
    echo " ready."
fi

mkdir -p "${INSPECT_LOG_DIR}"

# ── Start scaffold ────────────────────────────────────────────────────────────
start_scaffold() {
    pkill -f '[u]vicorn svc_scaffold.main:app' 2>/dev/null || true
    sleep 1
    cd "${REPO_ROOT}"
    "${VENV_PYTHON}" -m uvicorn svc_scaffold.main:app \
        --host "0.0.0.0" \
        --port "${SCAFFOLD_PORT}" \
        > "${STATE_DIR}/scaffold.stdout" \
        2> "${STATE_DIR}/scaffold.stderr" &
    SCAFFOLD_PID=$!
    printf "  scaffold starting..."
    i=0
    while ! check_scaffold 2>/dev/null; do
        kill -0 "${SCAFFOLD_PID}" 2>/dev/null || { echo " DIED"; cat "${STATE_DIR}/scaffold.stderr"; return 1; }
        i=$((i+1))
        [ $i -le 30 ] || { echo " TIMEOUT"; return 1; }
        printf "."
        sleep 1
    done
    echo " ready."
}

# ── Run one feature ──────────────────────────────────────────────────────────
run_feature() {
    FEAT="$1"

    # Skip if already done
    if [ -f "${DONE_DIR}/${FEAT}" ]; then
        echo ""
        echo "  [SKIP] ${FEAT} (already complete)"
        RESULTS="${RESULTS}SKIP ${FEAT}\n"
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "  ${FEAT}"
    echo "=========================================="

    # Export all FEATURE_* = 0, then override the target to 1
    for f in $FEATURES; do export "${f}=0"; done
    export "${FEAT}=1"

    start_scaffold || return 1

    RUN_STATUS=0
    "${REPO_ROOT}/.venv/bin/inspect" eval "${GAIA_TASK}" \
        --model "openai-api/gaia_model/${SCAFFOLD_MODEL_NAME}" \
        --sandbox "${GAIA_SANDBOX}" \
        --max-connections "${GAIA_MAX_CONNECTIONS}" \
        -T "split=${GAIA_SPLIT}" \
        -T "max_attempts=${GAIA_MAX_ATTEMPTS}" \
        --limit "${SAMPLE_LIMIT}" \
        --no-fail-on-error \
        2>&1 || RUN_STATUS=$?

    echo ""
    echo "  --- scaffold log (last 40 lines) ---"
    tail -40 "${STATE_DIR}/scaffold.stdout" 2>/dev/null || true

    pkill -f '[u]vicorn svc_scaffold.main:app' 2>/dev/null || true
    sleep 1

    if [ "${RUN_STATUS}" -eq 0 ]; then
        touch "${DONE_DIR}/${FEAT}"
        echo "  --> OK"
        RESULTS="${RESULTS}OK   ${FEAT}\n"
    else
        echo "  --> FAILED (exit ${RUN_STATUS})"
        RESULTS="${RESULTS}FAIL ${FEAT}\n"
    fi
}

# ── Iterate ──────────────────────────────────────────────────────────────────
for FEAT in $FEATURES; do
    [ -n "${FEAT}" ] || continue
    run_feature "${FEAT}" || RESULTS="${RESULTS}FAIL ${FEAT} (script error)\n"
done

echo ""
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
printf "%b" "${RESULTS}"
