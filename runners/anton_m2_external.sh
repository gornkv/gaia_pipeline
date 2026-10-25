#!/bin/sh
cd -- "$(dirname -- "$0")/.." && \
WORKER_NAME=anton_m2_external \
WORKER_BACKEND=external \
WORKER_MODEL=qwen3.7-flash \
WORKER_MODEL_CONTEXT_LIMIT=1000000 \
WORKER_PARALLEL=4 \
WORKER_GPU_NUM='' \
BACKEND_EXTERNAL_API_URL=https://api.aitunnel.ru/v1 \
exec .venv/bin/python -m worker.worker
