#!/bin/sh
set -aue

pkill -f '[v]llm' 2>/dev/null && sleep 2

LLAMA_SERVER="${REPO_ROOT}/_state/llama.cpp-cuda/cuda-12.8/llama-server"

if [ ! -f "${LLAMA_SERVER}" ]; then
  mkdir -p "${REPO_ROOT}/_state/downloads" "${REPO_ROOT}/_state/llama.cpp-cuda"
  curl -fL "https://github.com/ai-dock/llama.cpp-cuda/releases/download/b9568/llama.cpp-b9568-cuda-12.8-amd64.tar.gz" \
    -o "${REPO_ROOT}/_state/downloads/llama.cpp-b9568-cuda-12.8-amd64.tar.gz"
  rm -rf "${REPO_ROOT}/_state/llama.cpp-cuda"/*
  tar -xzf "${REPO_ROOT}/_state/downloads/llama.cpp-b9568-cuda-12.8-amd64.tar.gz" \
    -C "${REPO_ROOT}/_state/llama.cpp-cuda"
fi

chmod +x "${LLAMA_SERVER}"
[ -x "${LLAMA_SERVER}" ]

LD_LIBRARY_PATH="${REPO_ROOT}/_state/llama.cpp-cuda/cuda-12.8${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${LLAMA_SERVER}" \
  -hf "unsloth/Qwen3.5-9B-GGUF:Q4_K_M" \
  --alias "${BASE_MODEL_NAME}" \
  --host "0.0.0.0" \
  --port "18082" \
  --api-key "${BASE_MODEL_API_KEY}" \
  --ctx-size "131072" \
  --n-gpu-layers "999" \
  --threads "4" \
  --parallel "1" \
  --batch-size "512" \
  --ubatch-size "128" \
  --jinja \
  > "${REPO_ROOT}/_state/runner/${BASE_MODEL_RUNNER_TYPE}.stdout" \
  2> "${REPO_ROOT}/_state/runner/${BASE_MODEL_RUNNER_TYPE}.stderr" & pid=$!
echo "${pid}" >> "${PID_FILE}"

check_service() {
  "${VENV_PYTHON}" - <<'PY'
import os
from urllib.request import Request, urlopen

request = Request("http://127.0.0.1:18082/v1/models")
api_key = os.environ.get("BASE_MODEL_API_KEY")
if api_key:
    request.add_header("Authorization", f"Bearer {api_key}")
try:
    urlopen(request, timeout=5).close()
except Exception:
    raise SystemExit(1)
PY
}

while ! check_service; do
  kill -0 "$pid" 2>/dev/null || exit 1
  sleep 5
done
