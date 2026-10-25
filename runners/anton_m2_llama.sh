#!/bin/sh
cd -- "$(dirname -- "$0")/.." && \
WORKER_NAME=anton_m2_llama \
WORKER_BACKEND=llama \
WORKER_MODEL=bartowski/Qwen_Qwen3.5-9B-GGUF:Q4_K_M \
WORKER_MODEL_CONTEXT_LIMIT=16384 \
WORKER_PARALLEL=2 \
WORKER_GPU_NUM='' \
BACKEND_LLAMA_ACCELERATOR=metal \
BACKEND_LLAMA_KV_POOL_SIZE=65536 \
exec .venv/bin/python -m worker.worker
