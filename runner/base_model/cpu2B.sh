#!/bin/sh
set -aue

"${VENV_PYTHON}" -m pip install "llama-cpp-python[server]" "huggingface-hub"

"${VENV_PYTHON}" -m llama_cpp.server \
  --hf_model_repo_id "unsloth/Qwen3.5-2B-GGUF" \
  --model "*Q4_K_M*.gguf" \
  --model_alias "${BASE_MODEL_NAME}" \
  --host "0.0.0.0" \
  --port "${BASE_MODEL_PORT}" \
  --n_gpu_layers 0 \
  --n_ctx "65536" \
  --n_threads "1" \
  --chat_format chatml \
  > "${RUNNER_STATE_DIR}/${BASE_MODEL_RUNNER_TYPE}.stdout" \
  2> "${RUNNER_STATE_DIR}/${BASE_MODEL_RUNNER_TYPE}.stderr" & pid=$!
echo "${pid}" >> "${PID_FILE}"

check_service() {
  "${VENV_PYTHON}" - <<'PY'
import os
from urllib.request import urlopen

try:
    urlopen(f"http://127.0.0.1:{os.environ['BASE_MODEL_PORT']}/v1/models", timeout=5).close()
except Exception:
    raise SystemExit(1)
PY
}

while ! check_service; do
  kill -0 "$pid" 2>/dev/null || exit 1
  sleep 5
done
