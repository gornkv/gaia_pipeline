#!/bin/sh
set -aue

pkill -f '[l]lama_cpp.server' 2>/dev/null && sleep 2

"${VENV_PYTHON}" -m pip install "llama-cpp-python[server]" "huggingface-hub"

"${VENV_PYTHON}" -m llama_cpp.server \
  --hf_model_repo_id "unsloth/Qwen3.5-2B-GGUF" \
  --model "*Q4_K_M*.gguf" \
  --model_alias "${MODEL_NAME}" \
  --host "0.0.0.0" \
  --port "18080" \
  --n_gpu_layers 0 \
  --n_ctx "65536" \
  --n_threads "1" \
  --chat_format chatml \
  > "${REPO_ROOT}/_state/runner/${RUN_TASK_NAME}.stdout" \
  2> "${REPO_ROOT}/_state/runner/${RUN_TASK_NAME}.stderr" & pid=$!
echo "${pid}" >> "${PID_FILE}"

check_service() {
  "${VENV_PYTHON}" - <<'PY'
from urllib.request import urlopen

try:
    urlopen("http://127.0.0.1:18080/v1/models", timeout=5).close()
except Exception:
    raise SystemExit(1)
PY
}

while ! check_service; do
  kill -0 "$pid" 2>/dev/null || exit 1
  sleep 5
done
