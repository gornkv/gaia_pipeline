#!/bin/sh
# Start Gemma 4 as a standalone local OpenAI-compatible llama.cpp server.
# This intentionally does not start Inspect, Scaffold, a browser, or GAIA.
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

load_env() {
  [ -f "$1" ] || return 0

  cleaned_env="${TMPDIR:-/tmp}/gaia-pipeline-env.$$"
  trap 'rm -f "${cleaned_env:-}"' EXIT INT TERM
  tr -d '\r' < "$1" > "${cleaned_env}"
  set -a
  # shellcheck disable=SC1090
  . "${cleaned_env}"
  set +a
  rm -f "${cleaned_env}"
  cleaned_env=""
}

load_env "${ENV_FILE}"

# The model endpoint is always local. Inherited workstation proxy settings can
# divert even localhost traffic before NO_PROXY is considered by client code.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY OPENAI_PROXY
unset http_proxy https_proxy all_proxy openai_proxy
unset BROWSER_PROXY_URL BROWSER_PROXY_BYPASS BROWSER_PROXY_TEST_URL
unset INSPECT_BROWSER_PROXY INSPECT_BROWSER_PROXY_BYPASS
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export no_proxy="${NO_PROXY}"

MODEL_PATH="${GEMMA4_12B_IT_Q4KM_PATH:-}"
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
  MODEL_PATH="$1"
  shift
fi

if [ -z "${MODEL_PATH}" ]; then
  echo "Set GEMMA4_12B_IT_Q4KM_PATH in .env or pass the GGUF path as the first argument." >&2
  echo "Download: hf download hf://unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf" >&2
  exit 1
fi
if [ ! -f "${MODEL_PATH}" ]; then
  echo "Gemma 4 GGUF was not found: ${MODEL_PATH}" >&2
  echo "Download: hf download hf://unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf" >&2
  exit 1
fi

LLAMA_SERVER="${LOCAL_LLAMA_SERVER:-${LLAMA_SERVER_BIN:-llama-server}}"
if [ ! -x "${LLAMA_SERVER}" ]; then
  resolved_server="$(command -v "${LLAMA_SERVER}" 2>/dev/null || true)"
  if [ -n "${resolved_server}" ] && [ -x "${resolved_server}" ]; then
    LLAMA_SERVER="${resolved_server}"
  else
    echo "llama-server was not found: ${LLAMA_SERVER}" >&2
    exit 1
  fi
fi

HOST="${GEMMA4_HOST:-127.0.0.1}"
PORT="${GEMMA4_PORT:-8080}"
# GAIA requests reached 40K tokens in the prior local run, so 32K is unsafe.
CTX_SIZE="${GEMMA4_CTX_SIZE:-65536}"
GPU_LAYERS="${GEMMA4_GPU_LAYERS:-999}"
THREADS="${GEMMA4_THREADS:-4}"
PARALLEL="${GEMMA4_PARALLEL:-1}"
BATCH_SIZE="${GEMMA4_BATCH_SIZE:-512}"
UBATCH_SIZE="${GEMMA4_UBATCH_SIZE:-128}"
MODEL_ALIAS="${GEMMA4_MODEL_ALIAS:-gemma-4-12b-it-q4km}"

JINJA="${GEMMA4_JINJA:-auto}"
if [ "${JINJA}" = "auto" ]; then
  if "${LLAMA_SERVER}" --help 2>&1 | grep -q -- '--jinja'; then
    JINJA=1
  else
    JINJA=0
  fi
fi

set -- "${LLAMA_SERVER}" \
  --model "${MODEL_PATH}" \
  --alias "${MODEL_ALIAS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --ctx-size "${CTX_SIZE}" \
  --n-gpu-layers "${GPU_LAYERS}" \
  --threads "${THREADS}" \
  --parallel "${PARALLEL}" \
  --batch-size "${BATCH_SIZE}" \
  --ubatch-size "${UBATCH_SIZE}"

if [ "${JINJA}" = "1" ]; then
  set -- "$@" --jinja
fi

printf 'Starting Gemma 4 server at http://%s:%s/v1\n' "${HOST}" "${PORT}"
printf 'Model: %s\nContext: %s\nProxy: disabled\n' "${MODEL_PATH}" "${CTX_SIZE}"
exec "$@"
