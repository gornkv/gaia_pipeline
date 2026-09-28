#!/bin/sh
set -u
REPO_ROOT="/home/jovyan/nachevsky/experiments/gaia_pipeline"
cd "${REPO_ROOT}"; set -a; . ./.env; set +a
VENV="${REPO_ROOT}/.venv"; PY="${VENV}/bin/python"
export PATH="${VENV}/bin:${PATH}"
export HF_HOME="${REPO_ROOT}/_state/huggingface"
export PLAYWRIGHT_BROWSERS_PATH="${REPO_ROOT}/playwright-browsers"
export XDG_RUNTIME_DIR="${REPO_ROOT}/_state/run"; mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
export INSPECT_LOG_DIR="${REPO_ROOT}/_state/inspect-logs"
STATE="${REPO_ROOT}/_state/runner"; mkdir -p "$STATE"
export BASE_MODEL_API_BASE_URL="http://127.0.0.1:18081/v1"; export BASE_MODEL_API_KEY="EMPTY"; export BASE_MODEL_NAME="qwen3.5-9b-q4"
ALL_FLAGS="FEATURE_EXAMPLE FEATURE_BUDGET_TRACKER FEATURE_VOI FEATURE_SELF_CONSISTENCY FEATURE_SHORT_MAK FEATURE_ACON FEATURE_ADAPTIVE_BON FEATURE_CATTS FEATURE_BAVT FEATURE_RANKED_VOTING FEATURE_PICSAR FEATURE_MOB FEATURE_STRUCTURED_NOTES FEATURE_RESUM FEATURE_HIAGENT FEATURE_CONTEXT_COMPACTION FEATURE_COMPLEXITY_CONSISTENCY"
FEAT="${1:?usage: retry_feature.sh FEATURE_NAME}"
for f in $ALL_FLAGS; do export "$f=0"; done
export "${FEAT}=1"
echo "=== RETRY ${FEAT} ==="
check_vllm(){ curl -s --noproxy '*' -m5 -o /dev/null -H "Authorization: Bearer EMPTY" http://127.0.0.1:18081/v1/models; }
check_scaffold(){ curl -s --noproxy '*' -m5 -o /dev/null http://127.0.0.1:${SCAFFOLD_PORT}/health; }
if check_vllm; then echo "vLLM up"; else
  echo "starting vLLM (TP=4)..."; pkill -9 -f vllm 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; sleep 3
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
  nohup "${VENV}/bin/vllm" serve "Qwen/Qwen3.6-27B" --served-model-name "${BASE_MODEL_NAME}" --api-key EMPTY \
    --host 0.0.0.0 --port 18081 --dtype bfloat16 --max-model-len 131072 --tensor-parallel-size 2 \
    --max-num-seqs 64 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder --trust-remote-code > "${STATE}/vllm.stdout" 2> "${STATE}/vllm.stderr" &
  printf "waiting vLLM"; i=0; while ! check_vllm; do i=$((i+1)); [ $i -le 180 ] || { echo " TIMEOUT"; exit 1; }; printf "."; sleep 5; done; echo " ready"
fi
if [ -S "${XDG_RUNTIME_DIR}/inspect-tool-support.sock" ] && pgrep -f 'inspect-tool-support server' >/dev/null; then echo "browser up"; else
  rm -f "${XDG_RUNTIME_DIR}/inspect-tool-support.sock"
  nohup "${VENV}/bin/inspect-tool-support" server > "${STATE}/its-server.log" 2>&1 & sleep 6
fi
pkill -f '[u]vicorn svc_scaffold.main:app' 2>/dev/null; sleep 1
nohup "${PY}" -m uvicorn svc_scaffold.main:app --ws none --host 0.0.0.0 --port "${SCAFFOLD_PORT}" > "${STATE}/scaffold.stdout" 2> "${STATE}/scaffold.stderr" &
i=0; while ! check_scaffold; do i=$((i+1)); [ $i -le 30 ] || { echo scaffold FAIL; exit 1; }; sleep 1; done; echo "scaffold up"
export GAIA_MODEL_API_KEY="${SCAFFOLD_API_KEY}"; export GAIA_MODEL_BASE_URL="http://127.0.0.1:${SCAFFOLD_PORT}/v1"
LOG=$(ls -t "${INSPECT_LOG_DIR}"/20*.eval | head -1)
echo "retrying: $LOG"
"${VENV}/bin/inspect" eval-retry "$LOG" --max-connections 2 --max-samples 4
NEW=$(ls -t "${INSPECT_LOG_DIR}"/20*.eval | head -1)
cp "$NEW" "${INSPECT_LOG_DIR}/${FEAT}.eval"
echo "SAVED (86): ${INSPECT_LOG_DIR}/${FEAT}.eval"