#!/bin/sh
cd -- "$(dirname -- "$0")/.." && \
WORKER_NAME=anton_m2_llama \
WORKER_BACKEND=llama \
WORKER_MODEL=unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M \
WORKER_MODEL_CONTEXT_LIMIT=16384 \
WORKER_PARALLEL=2 \
WORKER_GPU_NUM='' \
BACKEND_LLAMA_ACCELERATOR=metal \
exec python3.12 -m worker
