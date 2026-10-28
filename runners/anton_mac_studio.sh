#!/bin/sh
cd -- "$(dirname -- "$0")/.." && \
WORKER_NAME=anton_mac_studio \
WORKER_BACKEND=llama \
WORKER_MODEL=bartowski/Qwen_Qwen3.5-9B-GGUF:Q4_K_M \
WORKER_MODEL_CONTEXT_LIMIT=262144 \
WORKER_PARALLEL=16 \
WORKER_GPU_NUM='' \
BACKEND_LLAMA_ACCELERATOR=metal \
BACKEND_LLAMA_BATCH_SIZE=8192 \
BACKEND_LLAMA_UBATCH_SIZE=256 \
exec /opt/homebrew/bin/python3.12 -m worker
