#!/bin/sh
set -u
REPO_ROOT="/home/jovyan/nachevsky/experiments/gaia_pipeline"
cd "${REPO_ROOT}"
set -a; . ./.env; set +a
VENV="${REPO_ROOT}/_state/venvs/gemma"; PY="${VENV}/bin/python"
export PATH="${VENV}/bin:${PATH}"
export HF_HOME="${REPO_ROOT}/_state/huggingface"
export PLAYWRIGHT_BROWSERS_PATH="${REPO_ROOT}/playwright-browsers"
export XDG_RUNTIME_DIR="${REPO_ROOT}/_state/run"; mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
export INSPECT_LOG_DIR="${REPO_ROOT}/_state/inspect-logs"
STATE="${REPO_ROOT}/_state/runner"; mkdir -p "$STATE"

# ---- Gemma-4-12B via vLLM ("old way") ----
GEMMA_MODEL="${GEMMA_MODEL:-unsloth/gemma-4-12b-it}"
GEMMA_MAXLEN="${GEMMA_MAXLEN:-131072}"
GEMMA_TOOL_PARSER="${GEMMA_TOOL_PARSER:-pythonic}"
export BASE_MODEL_API_BASE_URL="http://127.0.0.1:18081/v1"
export BASE_MODEL_API_KEY="EMPTY"; export BASE_MODEL_NAME="gemma-4-12b-it"

ALL_FLAGS="FEATURE_EXAMPLE FEATURE_BUDGET_TRACKER FEATURE_VOI FEATURE_SELF_CONSISTENCY FEATURE_SHORT_MAK FEATURE_ACON FEATURE_ADAPTIVE_BON FEATURE_CATTS FEATURE_BAVT FEATURE_RANKED_VOTING FEATURE_PICSAR FEATURE_MOB FEATURE_STRUCTURED_NOTES FEATURE_RESUM FEATURE_HIAGENT FEATURE_CONTEXT_COMPACTION FEATURE_COMPLEXITY_CONSISTENCY"

# feature = 1st arg (recommended), else whatever is =1 in .env
if [ -n "${1:-}" ]; then
  for f in $ALL_FLAGS; do export "$f=0"; done
  [ "$1" != "baseline" ] && export "$1=1"
  FEAT="$1"
else
  FEAT="$(env | sed -n 's/^\(FEATURE_[A-Z_]*\)=1$/\1/p' | head -1)"; [ -n "$FEAT" ] || FEAT="baseline"
fi
FEAT="gemma_${FEAT}"
echo "=== MODEL: ${GEMMA_MODEL} | PARSER: ${GEMMA_TOOL_PARSER} | FEATURE: ${FEAT} ==="

check_vllm(){ curl -s --noproxy '*' -m 5 -o /dev/null -H "Authorization: Bearer EMPTY" http://127.0.0.1:18081/v1/models; }
check_scaffold(){ curl -s --noproxy '*' -m 5 -o /dev/null http://127.0.0.1:${SCAFFOLD_PORT}/health; }

# 1) vLLM serving Gemma (ALWAYS restart so we don't reuse a Qwen server on 18081)
echo "starting vLLM for Gemma (tensor-parallel 2)..."
echo skip-pkill-shared-node; sleep 1
NVLIBS="$("${PY}" - <<'PYX'
from pathlib import Path; import site
p=[]
for b in site.getsitepackages():
    d=Path(b)/"nvidia"
    if d.is_dir(): p+=[str(x) for x in d.glob("*/lib") if x.is_dir()]
print(":".join(p))
PYX
)"
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
nohup "${VENV}/bin/vllm" serve "${GEMMA_MODEL}" \
  --served-model-name "${BASE_MODEL_NAME}" --api-key EMPTY \
  --host 0.0.0.0 --port 18081 --dtype bfloat16 --max-model-len "${GEMMA_MAXLEN}" \
  --tensor-parallel-size 1 --max-num-seqs 64 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser "${GEMMA_TOOL_PARSER}" \
  --trust-remote-code > "${STATE}/vllm.stdout" 2> "${STATE}/vllm.stderr" &
printf "waiting vLLM (first run downloads ~24GB, be patient)"; i=0
while ! check_vllm; do i=$((i+1)); [ $i -le 360 ] || { echo " TIMEOUT (see ${STATE}/vllm.stderr)"; exit 1; }; printf "."; sleep 5; done; echo " ready."

# 2) browser tool-support server (once, self-healing)
if [ -S "${XDG_RUNTIME_DIR}/inspect-tool-support.sock" ] && pgrep -f 'inspect-tool-support server' >/dev/null; then
  echo "browser server up"
else
  echo "starting browser server..."; rm -f "${XDG_RUNTIME_DIR}/inspect-tool-support.sock"
  nohup "${VENV}/bin/inspect-tool-support" server > "${STATE}/its-server.log" 2>&1 & sleep 6
fi

# 3) restart scaffold so it reads the FEATURE flags + new BASE_MODEL_NAME
pkill -f '[u]vicorn svc_scaffold.main:app' 2>/dev/null; sleep 1
nohup "${PY}" -m uvicorn svc_scaffold.main:app --ws none --host 0.0.0.0 --port "${SCAFFOLD_PORT}" \
  > "${STATE}/scaffold.stdout" 2> "${STATE}/scaffold.stderr" &
i=0; while ! check_scaffold; do i=$((i+1)); [ $i -le 30 ] || { echo "scaffold FAILED"; cat "${STATE}/scaffold.stderr"; exit 1; }; sleep 1; done
echo "scaffold up"

# 4) run the eval
export GAIA_MODEL_API_KEY="${SCAFFOLD_API_KEY}"; export GAIA_MODEL_BASE_URL="http://127.0.0.1:${SCAFFOLD_PORT}/v1"
"${VENV}/bin/inspect" eval "${GAIA_TASK}" \
  --model "openai-api/gaia_model/${SCAFFOLD_MODEL_NAME}" --sandbox "${GAIA_SANDBOX}" \
  --max-connections "${GAIA_MAX_CONNECTIONS}" --max-samples "${GAIA_MAX_SAMPLES:-4}" \
  -T "split=${GAIA_SPLIT}" -T "max_attempts=${GAIA_MAX_ATTEMPTS}" \
  --time-limit "${GAIA_SAMPLE_TIME_LIMIT}" --message-limit "${GAIA_MESSAGE_LIMIT}" \
  --no-fail-on-error

# 5) save a copy named by feature (gemma_ prefix keeps it separate from Qwen)
LATEST="$(ls -t "${INSPECT_LOG_DIR}"/20*.eval 2>/dev/null | head -1)"
[ -n "$LATEST" ] && cp "$LATEST" "${INSPECT_LOG_DIR}/${FEAT}.eval" && echo "SAVED: ${INSPECT_LOG_DIR}/${FEAT}.eval"
