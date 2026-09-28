#!/bin/sh
set -u
REPO_ROOT="/home/jovyan/nachevsky/experiments/gaia_pipeline"
cd "${REPO_ROOT}"; set -a; . ./.env; set +a
VENV="${REPO_ROOT}/_state/venvs/gemma"; PY="${VENV}/bin/python"; SP="${VENV}/lib/python3.12/site-packages"
export PATH="${VENV}/bin:${PATH}"
export HF_HOME="${REPO_ROOT}/_state/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
export NO_PROXY="*" no_proxy="*"
STATE="${REPO_ROOT}/_state/runner"; mkdir -p "$STATE"
MODEL="openai/gpt-oss-120b"
export CUDA_VISIBLE_DEVICES=2; unset NVCC_APPEND_FLAGS CPATH
NVLIBS="$("${PY}" - <<PYX
from pathlib import Path; import site
p=[]
for b in site.getsitepackages():
    d=Path(b)/"nvidia"
    if d.is_dir(): p+=[str(x) for x in d.glob("*/lib") if x.is_dir()]
print(":".join(p))
PYX
)"
export LD_LIBRARY_PATH="${SP}/torch/lib:${NVLIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "serving gpt-oss card2:18090"
exec "${VENV}/bin/vllm" serve "${MODEL}" --served-model-name gpt-oss-120b --api-key EMPTY \
  --host 0.0.0.0 --port 18090 --max-model-len 32768 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 --trust-remote-code
