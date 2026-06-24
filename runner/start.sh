#!/bin/sh
set -aue

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

load_env() {
  ENV_FILE="${TMPDIR:-/tmp}/gaia-pipeline-env.$$"
  tr -d '\r' < "$1" > "${ENV_FILE}"
  . "${ENV_FILE}"
}

load_env "${REPO_ROOT}/.env"

# Ensure /shared_files exists and is owned by the current user (fix inspect sandbox setup)
mkdir -p /shared_files 2>/dev/null || sudo mkdir -p /shared_files || true
chown "$(id -u):$(id -g)" /shared_files 2>/dev/null || sudo chown "$(id -u):$(id -g)" /shared_files 2>/dev/null || true

mkdir -p "${REPO_ROOT}/_state/runner"
if [ "${START_LOGGING:-}" != "1" ]; then
  rm -rf "${REPO_ROOT}/_state/runner"/*
  START_STDOUT_PIPE="${TMPDIR:-/tmp}/gaia-pipeline.stdout.pipe.$$"
  START_STDERR_PIPE="${TMPDIR:-/tmp}/gaia-pipeline.stderr.pipe.$$"
  mkfifo "${START_STDOUT_PIPE}" "${START_STDERR_PIPE}"
  START_LOGGING=1
  tee "${REPO_ROOT}/_state/runner/start.stdout" < "${START_STDOUT_PIPE}" &
  tee "${REPO_ROOT}/_state/runner/start.stderr" >&2 < "${START_STDERR_PIPE}" &
  exec sh "$0" "$@" > "${START_STDOUT_PIPE}" 2> "${START_STDERR_PIPE}"
fi

VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
HF_HOME="${REPO_ROOT}/_state/huggingface"
INSPECT_LOG_DIR="${REPO_ROOT}/_state/inspect-logs"
PLAYWRIGHT_BROWSERS_PATH="${REPO_ROOT}/playwright-browsers"
PATH="${REPO_ROOT}/.venv/bin:${PATH}"
PID_FILE="${REPO_ROOT}/_state/runner/pids"

cd "${REPO_ROOT}"
mkdir -p \
  "${HF_HOME}" \
  "${INSPECT_LOG_DIR}" \
  "${PLAYWRIGHT_BROWSERS_PATH}" \
  "${REPO_ROOT}/_state/runner"
: > "${PID_FILE}"

cleanup() {
  status=$?
  if [ -f "${PID_FILE}" ]; then
    while IFS= read -r pid; do
      [ -n "${pid}" ] || continue
      kill "${pid}" 2>/dev/null || :
    done < "${PID_FILE}"
  fi
  rm -f "${START_STDOUT_PIPE:-}" "${START_STDERR_PIPE:-}"
  exit "${status}"
}

trap cleanup EXIT
trap 'exit 1' INT TERM

create_venv() {
  if "${VENV_PYTHON}" -m pip --version >/dev/null 2>&1; then
    return
  fi

  if "${PYTHON_BIN:-python3}" -m venv "${REPO_ROOT}/.venv"; then
    return
  fi

  rm -rf "${REPO_ROOT}/.venv"
  "${PYTHON_BIN:-python3}" -m venv --system-site-packages --without-pip "${REPO_ROOT}/.venv"
  "${VENV_PYTHON}" -m pip install --upgrade pip wheel setuptools
}

check_evironment() {
  "${VENV_PYTHON}" - <<'PY'
from importlib.metadata import version
from shutil import which

import fastapi  # noqa: F401
import httpx  # noqa: F401
import inspect_ai  # noqa: F401
import inspect_evals.gaia  # noqa: F401
import openai  # noqa: F401
import playwright  # noqa: F401
import uvicorn  # noqa: F401

print(f"inspect-ai={version('inspect-ai')}")
print(f"inspect-evals={version('inspect-evals')}")

if which("inspect-tool-support") is None:
    raise RuntimeError("inspect-tool-support executable is not available on PATH")
PY
}

install_environment() {
  create_venv
  "${VENV_PYTHON}" -m pip install \
    pip wheel setuptools "inspect-evals[gaia]" inspect-tool-support openai playwright \
    -r "${REPO_ROOT}/svc_scaffold/requirement.txt"
  "${REPO_ROOT}/.venv/bin/inspect-tool-support" post-install
  "${VENV_PYTHON}" -m playwright install chromium
  check_evironment
}

if ! check_evironment >/dev/null 2>&1; then
  install_environment
fi

load_env "${REPO_ROOT}/runner/base_model/${BASE_MODEL_RUNNER_TYPE}.env"
sh "${REPO_ROOT}/runner/base_model/${BASE_MODEL_RUNNER_TYPE}.sh"
sh "${REPO_ROOT}/runner/scaffold.sh"
sh "${REPO_ROOT}/runner/benchmark.sh"
