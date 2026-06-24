#!/bin/sh
set -aue

pkill -f '[v]llm' 2>/dev/null && sleep 2

# Detect GPU compute capability; default to 7.5 (prebuilt-safe) if unavailable.
GPU_COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')"
GPU_COMPUTE_CAP="${GPU_COMPUTE_CAP:-7.5}"
# Convert "7.0" → 70, "8.6" → 86, etc. for integer comparison.
GPU_SM="$(echo "${GPU_COMPUTE_CAP}" | awk -F. '{printf "%d%s", $1, $2}')"

if [ "${GPU_SM}" -lt 75 ]; then
  # Prebuilt b9568 requires sm >= 75; build from source for this GPU (e.g. V100 = sm_70).
  LLAMA_SRC_DIR="${REPO_ROOT}/_state/llama.cpp-cuda/src"
  LLAMA_BUILD_DIR="${REPO_ROOT}/_state/llama.cpp-cuda/build"
  LLAMA_SERVER="${LLAMA_BUILD_DIR}/bin/llama-server"
  LLAMA_TAG="b9568"

  if [ ! -f "${LLAMA_SERVER}" ]; then
    if ! command -v cmake >/dev/null 2>&1; then
      sudo apt-get install -y cmake
    fi

    NVCC_BIN="$(command -v nvcc 2>/dev/null || find /usr/local/cuda*/bin -name nvcc 2>/dev/null | sort -rV | head -1)"
    CUDA_TOOLKIT="$(dirname "$(dirname "${NVCC_BIN}")")"

    mkdir -p "${REPO_ROOT}/_state/llama.cpp-cuda"

    if [ ! -d "${LLAMA_SRC_DIR}/.git" ]; then
      git clone --depth=1 --branch "${LLAMA_TAG}" \
        https://github.com/ggerganov/llama.cpp.git "${LLAMA_SRC_DIR}"
    fi

    cmake -S "${LLAMA_SRC_DIR}" -B "${LLAMA_BUILD_DIR}" \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_COMPILER="${CUDA_TOOLKIT}/bin/nvcc" \
      -DCMAKE_CUDA_ARCHITECTURES="${GPU_SM}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_OPENSSL=ON

    cmake --build "${LLAMA_BUILD_DIR}" --target llama-server -j "$(nproc)"

    # Build thin wrapper plugin that exposes ggml_backend_init expected by the plugin loader.
    # libggml-cuda.so has ggml_backend_cuda_reg but not ggml_backend_init.
    printf 'typedef struct ggml_backend_reg * ggml_backend_reg_t;\nextern ggml_backend_reg_t ggml_backend_cuda_reg(void);\nggml_backend_reg_t ggml_backend_init(void) { return ggml_backend_cuda_reg(); }\n' \
      > "${LLAMA_BUILD_DIR}/ggml_cuda_plugin.c"
    gcc -shared -fPIC -O2 "${LLAMA_BUILD_DIR}/ggml_cuda_plugin.c" \
      -L"${LLAMA_BUILD_DIR}/bin" -lggml-cuda \
      -Wl,-rpath,"${LLAMA_BUILD_DIR}/bin" \
      -o "${LLAMA_BUILD_DIR}/bin/ggml-cuda-plugin.so"
  fi

  LD_LIBRARY_PATH="${LLAMA_BUILD_DIR}/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  GGML_BACKEND_PATH="${LLAMA_BUILD_DIR}/bin/ggml-cuda-plugin.so"
else
  # Use prebuilt binary (sm >= 75: RTX 2000+, A100, H100, etc.)
  LLAMA_SERVER="${REPO_ROOT}/_state/llama.cpp-cuda/cuda-12.8/llama-server"

  if [ ! -f "${LLAMA_SERVER}" ]; then
    mkdir -p "${REPO_ROOT}/_state/downloads" "${REPO_ROOT}/_state/llama.cpp-cuda"
    curl -fL "https://github.com/ai-dock/llama.cpp-cuda/releases/download/b9568/llama.cpp-b9568-cuda-12.8-amd64.tar.gz" \
      -o "${REPO_ROOT}/_state/downloads/llama.cpp-b9568-cuda-12.8-amd64.tar.gz"
    rm -rf "${REPO_ROOT}/_state/llama.cpp-cuda"/*
    tar -xzf "${REPO_ROOT}/_state/downloads/llama.cpp-b9568-cuda-12.8-amd64.tar.gz" \
      -C "${REPO_ROOT}/_state/llama.cpp-cuda"
  fi

  LD_LIBRARY_PATH="${REPO_ROOT}/_state/llama.cpp-cuda/cuda-12.8${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

chmod +x "${LLAMA_SERVER}"
[ -x "${LLAMA_SERVER}" ]

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
  -fit off \
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
