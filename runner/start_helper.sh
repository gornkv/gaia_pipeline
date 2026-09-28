#!/bin/sh
# Serve the strong HELPER model (gpt-oss-120B) on a dedicated GPU + port 18082,
# alongside the eval vLLM. Point the scaffold at it via .env:
#   HELPER_API_BASE_URL=http://127.0.0.1:18082/v1
#   HELPER_MODEL=gpt-oss-120b
#   HELPER_API_KEY=EMPTY
#   HELPER_REASONING_EFFORT=low
set -u
REPO_ROOT="/home/jovyan/nachevsky/experiments/gaia_pipeline"
cd "${REPO_ROOT}"
VENV="${REPO_ROOT}/.venv"; PY="${VENV}/bin/python"
export PATH="${VENV}/bin:${PATH}"
export HF_HOME="${REPO_ROOT}/_state/huggingface"
STATE="${REPO_ROOT}/_state/runner"; mkdir -p "$STATE"

HELPER_MODEL_ID="${HELPER_MODEL_ID:-openai/gpt-oss-120b}"
HELPER_SERVED="${HELPER_SERVED:-gpt-oss-120b}"
HELPER_GPUS="${HELPER_GPUS:-3}"          # which card(s) for the helper, e.g. 3 or 2,3
HELPER_PORT="${HELPER_PORT:-18082}"
HELPER_TP="${HELPER_TP:-1}"
HELPER_MAXLEN="${HELPER_MAXLEN:-65536}"

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

echo "starting HELPER ${HELPER_MODEL_ID} on GPU(s) ${HELPER_GPUS}, port ${HELPER_PORT} ..."
CUDA_VISIBLE_DEVICES="${HELPER_GPUS}" nohup "${VENV}/bin/vllm" serve "${HELPER_MODEL_ID}" \
  --served-model-name "${HELPER_SERVED}" --api-key EMPTY \
  --host 0.0.0.0 --port "${HELPER_PORT}" --max-model-len "${HELPER_MAXLEN}" \
  --tensor-parallel-size "${HELPER_TP}" --gpu-memory-utilization 0.92 \
  --trust-remote-code > "${STATE}/helper.stdout" 2> "${STATE}/helper.stderr" &

printf "waiting helper"; i=0
while ! curl -s -m 5 -o /dev/null -H "Authorization: Bearer EMPTY" "http://127.0.0.1:${HELPER_PORT}/v1/models"; do
  i=$((i+1)); [ $i -le 360 ] || { echo " TIMEOUT (see ${STATE}/helper.stderr)"; exit 1; }; printf "."; sleep 5
done
echo " ready. Helper at http://127.0.0.1:${HELPER_PORT}/v1 (model ${HELPER_SERVED})"
