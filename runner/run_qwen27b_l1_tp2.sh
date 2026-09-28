#!/bin/sh
set -u
REPO_ROOT="/home/jovyan/nachevsky/experiments/gaia_pipeline"
cd "${REPO_ROOT}"; set -a; . ./.env; set +a
export GAIA_TASK="inspect_evals/gaia_level1"  # 27B canon = L1 (decoupled from .env)
VENV="${REPO_ROOT}/_state/venvs/gemma"; PY="${VENV}/bin/python"
export PATH="${VENV}/bin:${PATH}"
export HF_HOME="${REPO_ROOT}/_state/huggingface"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
export NO_PROXY="*" no_proxy="*"
export PLAYWRIGHT_BROWSERS_PATH="${REPO_ROOT}/playwright-browsers"
STATE="${REPO_ROOT}/_state/runner"; mkdir -p "$STATE"
export BASE_MODEL_API_KEY="EMPTY"; export BASE_MODEL_NAME="qwen3.6-27b"
GEMMA_MODEL="${GEMMA_MODEL:-Qwen/Qwen3.6-27B}"
GEMMA_TOOL_PARSER="${GEMMA_TOOL_PARSER:-qwen3_coder}"

ARG="${1:?usage: run_gemma_lane.sh baseline|FEATURE a|b [tag]}"
LANE="${2:?lane a or b}"
TAG="${3:-}"
case "$LANE" in
  a) CARD="0,1"; VPORT=18081; SPORT=19080; XDG="${REPO_ROOT}/_state/run";   LOGDIR="${REPO_ROOT}/_state/inspect-logs" ;;
  b) CARD="2,3"; VPORT=18082; SPORT=19081; XDG="${REPO_ROOT}/_state/run_b"; LOGDIR="${REPO_ROOT}/_state/inspect-logs-b" ;;
  a2) CARD="0,1"; VPORT=18081; SPORT=19084; XDG="${REPO_ROOT}/_state/run_a2"; LOGDIR="${REPO_ROOT}/_state/inspect-logs-a2" ;;
  b2) CARD="2,3"; VPORT=18082; SPORT=19085; XDG="${REPO_ROOT}/_state/run_b2"; LOGDIR="${REPO_ROOT}/_state/inspect-logs-b2" ;;
  c) CARD=0; VPORT=18083; SPORT=19082; XDG="${REPO_ROOT}/_state/run_c"; LOGDIR="${REPO_ROOT}/_state/inspect-logs-c" ;;
  d) CARD=1; VPORT=18084; SPORT=19083; XDG="${REPO_ROOT}/_state/run_d"; LOGDIR="${REPO_ROOT}/_state/inspect-logs-d" ;;
  *) echo "lane must be a b c or d"; exit 1 ;;
esac
mkdir -p "$XDG" "$LOGDIR"; chmod 700 "$XDG"
export XDG_RUNTIME_DIR="$XDG"; export INSPECT_LOG_DIR="$LOGDIR"
export BASE_MODEL_API_BASE_URL="http://127.0.0.1:${VPORT}/v1"
export SCAFFOLD_PORT="$SPORT"

ALL_FLAGS="FEATURE_EXAMPLE FEATURE_BUDGET_TRACKER FEATURE_VOI FEATURE_SELF_CONSISTENCY FEATURE_SHORT_MAK FEATURE_ACON FEATURE_ADAPTIVE_BON FEATURE_CATTS FEATURE_BAVT FEATURE_RANKED_VOTING FEATURE_PICSAR FEATURE_MOB FEATURE_STRUCTURED_NOTES FEATURE_RESUM FEATURE_HIAGENT FEATURE_CONTEXT_COMPACTION FEATURE_COMPLEXITY_CONSISTENCY FEATURE_THINK_IN_TEXT FEATURE_FORCE_NOTES"
for f in $ALL_FLAGS; do export "$f=0"; done
[ "$ARG" != "baseline" ] && export "${ARG}=1"
[ "${TIT:-0}" = "1" ] && export FEATURE_THINK_IN_TEXT=1
[ "${FORCE:-0}" = "1" ] && export FEATURE_FORCE_NOTES=1
FEAT="qwen27b_${ARG}${TAG:+_${TAG}}"
echo "=== LANE ${LANE} card${CARD} vllm:${VPORT} scaffold:${SPORT} | RUN ${FEAT} | parser ${GEMMA_TOOL_PARSER} ==="

check_vllm(){ curl -s --noproxy "*" -m5 -o /dev/null -H "Authorization: Bearer EMPTY" http://127.0.0.1:${VPORT}/v1/models; }
check_scaffold(){ curl -s --noproxy "*" -m5 -o /dev/null http://127.0.0.1:${SPORT}/health; }

if check_vllm; then echo "vLLM up (reuse) on ${VPORT}"; else
  echo "serving Qwen3.6-27B TP=1 on card ${CARD} port ${VPORT}"
  export CUDA_VISIBLE_DEVICES=$CARD; unset NVCC_APPEND_FLAGS CPATH
  NVLIBS="$("${PY}" - <<PYX
from pathlib import Path; import site
p=[]
for b in site.getsitepackages():
    d=Path(b)/"nvidia"
    if d.is_dir(): p+=[str(x) for x in d.glob("*/lib") if x.is_dir()]
print(":".join(p))
PYX
)"
  export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  nohup "${VENV}/bin/vllm" serve "${GEMMA_MODEL}" --served-model-name "${BASE_MODEL_NAME}" --api-key EMPTY \
    --host 0.0.0.0 --port ${VPORT} --dtype bfloat16 --max-model-len 131072 --tensor-parallel-size 2 \
    --max-num-seqs 64 --gpu-memory-utilization 0.92 --enable-auto-tool-choice \
    --tool-call-parser "${GEMMA_TOOL_PARSER}" --reasoning-parser qwen3 --trust-remote-code > "${STATE}/vllm_${LANE}.stdout" 2> "${STATE}/vllm_${LANE}.stderr" &
  printf "waiting vLLM"; i=0; while ! check_vllm; do i=$((i+1)); [ $i -le 360 ] || { echo " TIMEOUT"; exit 1; }; printf "."; sleep 5; done; echo " ready"
fi

# browser tool-support server bound to THIS lane XDG socket
lane_its_alive() {
  for p in $(ls /proc 2>/dev/null | grep -E "^[0-9]+$"); do
    c=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null)
    case "$c" in *inspect-tool-support*) ;; *) continue ;; esac
    tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -q "^XDG_RUNTIME_DIR=${XDG}$" && return 0
  done
  return 1
}
if [ -S "${XDG}/inspect-tool-support.sock" ] && lane_its_alive; then echo "browser up (lane ${LANE})"; else
  rm -f "${XDG}/inspect-tool-support.sock"
  nohup env XDG_RUNTIME_DIR="$XDG" PLAYWRIGHT_BROWSERS_PATH="$PLAYWRIGHT_BROWSERS_PATH" "${VENV}/bin/inspect-tool-support" server > "${STATE}/its_${LANE}.log" 2>&1 & sleep 6
fi

# restart THIS lane scaffold (match its port in cmdline; pkill broken on node)
for p in $(ls /proc 2>/dev/null | grep -E "^[0-9]+$"); do c=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null); echo "$c" | grep -q "svc_scaffold.main:app" && echo "$c" | grep -q "port ${SPORT}" && kill -9 $p 2>/dev/null; done; sleep 1
nohup "${PY}" -m uvicorn svc_scaffold.main:app --ws none --host 0.0.0.0 --port "${SPORT}" > "${STATE}/scaffold_${LANE}.stdout" 2> "${STATE}/scaffold_${LANE}.stderr" &
i=0; while ! check_scaffold; do i=$((i+1)); [ $i -le 30 ] || { echo scaffold FAIL; cat "${STATE}/scaffold_${LANE}.stderr"; exit 1; }; sleep 1; done; echo "scaffold up (lane ${LANE})"

export GAIA_MODEL_API_KEY="${SCAFFOLD_API_KEY}"; export GAIA_MODEL_BASE_URL="http://127.0.0.1:${SPORT}/v1"
"${VENV}/bin/inspect" eval "${GAIA_TASK}" \
  --model "openai-api/gaia_model/${SCAFFOLD_MODEL_NAME}" --sandbox "${GAIA_SANDBOX}" \
  --max-connections "${GAIA_MAX_CONNECTIONS}" --max-samples "${GAIA_MAX_SAMPLES:-4}" \
  -T "split=${GAIA_SPLIT}" -T "max_attempts=${GAIA_MAX_ATTEMPTS}" \
  --time-limit "${GAIA_SAMPLE_TIME_LIMIT}" --message-limit "${GAIA_MESSAGE_LIMIT}" --no-fail-on-error
LATEST="$(ls -t "${LOGDIR}"/20*.eval 2>/dev/null | head -1)"
[ -n "$LATEST" ] && cp "$LATEST" "${LOGDIR}/${FEAT}.eval" && echo "SAVED: ${LOGDIR}/${FEAT}.eval"
