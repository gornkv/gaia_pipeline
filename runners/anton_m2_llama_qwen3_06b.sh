#!/bin/sh
cd -- "$(dirname -- "$0")/.." && \
WORKER_NAME=anton_m2_llama_qwen3_06b \
WORKER_BACKEND=llama \
WORKER_MODEL=unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
WORKER_MODEL_CONTEXT_LIMIT=16384 \
WORKER_PARALLEL=2 \
WORKER_GPU_NUM='' \
BACKEND_LLAMA_ACCELERATOR=metal \
exec .venv/bin/python -m worker.worker
