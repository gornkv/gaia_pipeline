#!/bin/sh
set -aue

pkill -f '[v]llm' 2>/dev/null && sleep 2

"${VENV_PYTHON}" -m pip install "vllm>=0.22.1" ninja

NVIDIA_LIB_PATHS="$("${VENV_PYTHON}" - <<'PY'
from pathlib import Path
import site

paths = []
for base in site.getsitepackages():
    nvidia_dir = Path(base) / "nvidia"
    if nvidia_dir.is_dir():
        paths.extend(str(path) for path in nvidia_dir.glob("*/lib") if path.is_dir())
print(":".join(paths))
PY
)"
LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PATH="${REPO_ROOT}/.venv/bin:${PATH}"
VLLM_USE_TRITON_AWQ=1

"${REPO_ROOT}/.venv/bin/vllm" serve "QuantTrio/Qwen3.5-9B-AWQ" \
  --served-model-name "${BASE_MODEL_NAME}" \
  --api-key "${BASE_MODEL_API_KEY}" \
  --host "0.0.0.0" \
  --port "18081" \
  --dtype float16 \
  --language-model-only \
  --max-model-len "8192" \
  --max-num-seqs "1" \
  --max-num-batched-tokens "1024" \
  --gpu-memory-utilization "0.90" \
  --block-size "32" \
  --disable-custom-all-reduce \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --enforce-eager \
  > "${REPO_ROOT}/_state/runner/${BASE_MODEL_RUNNER_TYPE}.stdout" \
  2> "${REPO_ROOT}/_state/runner/${BASE_MODEL_RUNNER_TYPE}.stderr" & pid=$!
echo "${pid}" >> "${PID_FILE}"

check_service() {
  "${VENV_PYTHON}" - <<'PY'
import os
from urllib.request import Request, urlopen

request = Request("http://127.0.0.1:18081/v1/models")
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
