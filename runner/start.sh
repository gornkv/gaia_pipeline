#!/bin/sh
set -aue

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

load_env() {
  ENV_FILE="${TMPDIR:-/tmp}/gaia-pipeline-env.$$"
  tr -d '\r' < "$1" > "${ENV_FILE}"
  . "${ENV_FILE}"
}

load_env "${REPO_ROOT}/.env"

FEATURE_FLAGS="FEATURE_EXAMPLE FEATURE_ACON FEATURE_ADAPTIVE_BON FEATURE_BAVT FEATURE_BUDGET_TRACKER FEATURE_CATTS FEATURE_COMPLEXITY_CONSISTENCY FEATURE_CONTEXT_COMPACTION FEATURE_HIAGENT FEATURE_MOB FEATURE_PICSAR FEATURE_RANKED_VOTING FEATURE_RESUM FEATURE_SELF_CONSISTENCY FEATURE_SHORT_MAK FEATURE_STRUCTURED_NOTES FEATURE_VOI"
FEATURE_NAME="${1:-}"
LANE="${2:-}"
RUN_TAG="${3:-}"

if [ -n "${FEATURE_NAME}" ]; then
  case " ${FEATURE_FLAGS} baseline " in
    *" ${FEATURE_NAME} "*) ;;
    *) echo "Unknown feature: ${FEATURE_NAME}" >&2; exit 2 ;;
  esac
  for feature_flag in ${FEATURE_FLAGS}; do export "${feature_flag}=0"; done
  [ "${FEATURE_NAME}" = "baseline" ] || export "${FEATURE_NAME}=1"
else
  FEATURE_NAME="$(env | sed -n 's/^\(FEATURE_[A-Z_]*\)=1$/\1/p' | head -1)"
  FEATURE_NAME="${FEATURE_NAME:-baseline}"
fi

case "${RUN_TAG}" in *[!A-Za-z0-9._-]*) echo "Invalid tag: ${RUN_TAG}" >&2; exit 2 ;; esac
case "${LANE}" in
  "") LANE_SUFFIX="" ;;
  a) CUDA_VISIBLE_DEVICES=3; BASE_MODEL_PORT=18081; SCAFFOLD_PORT=19080; LANE_SUFFIX="" ;;
  b) CUDA_VISIBLE_DEVICES=2; BASE_MODEL_PORT=18082; SCAFFOLD_PORT=19081; LANE_SUFFIX="_b" ;;
  c) CUDA_VISIBLE_DEVICES=0; BASE_MODEL_PORT=18083; SCAFFOLD_PORT=19082; LANE_SUFFIX="_c" ;;
  d) CUDA_VISIBLE_DEVICES=1; BASE_MODEL_PORT=18084; SCAFFOLD_PORT=19083; LANE_SUFFIX="_d" ;;
  *) echo "Lane must be a, b, c, or d" >&2; exit 2 ;;
esac

RUNNER_STATE_DIR="${REPO_ROOT}/_state/runner${LANE_SUFFIX}"
INSPECT_LOG_DIR="${REPO_ROOT}/_state/inspect-logs${LANE_SUFFIX}"
XDG_RUNTIME_DIR="${REPO_ROOT}/_state/run${LANE_SUFFIX}"
GAIA_MODEL_BASE_URL="http://127.0.0.1:${SCAFFOLD_PORT}/v1"
PID_FILE="${RUNNER_STATE_DIR}/pids"

cleanup_previous_run() {
  [ -f "${PID_FILE}" ] || return
  killed=0
  while IFS= read -r old_pid; do
    case "${old_pid}" in ""|*[!0-9]*) continue ;; esac
    command="$(ps -p "${old_pid}" -o command= 2>/dev/null || true)"
    case "${command}" in
      *"${REPO_ROOT}"*|*"inspect-tool-support server"*|*"svc_scaffold.main:app"*)
        kill "${old_pid}" 2>/dev/null && killed=1 || true
        ;;
    esac
  done < "${PID_FILE}"
  [ "${killed}" = "0" ] || sleep 2
}

# Ensure /shared_files exists and is owned by the current user (fix inspect sandbox setup)
mkdir -p /shared_files 2>/dev/null || sudo mkdir -p /shared_files || true
chown "$(id -u):$(id -g)" /shared_files 2>/dev/null || sudo chown "$(id -u):$(id -g)" /shared_files 2>/dev/null || true

mkdir -p "${RUNNER_STATE_DIR}"
if [ "${START_LOGGING:-}" != "1" ]; then
  cleanup_previous_run
  find "${RUNNER_STATE_DIR}" -mindepth 1 -maxdepth 1 -delete
  START_STDOUT_PIPE="${TMPDIR:-/tmp}/gaia-pipeline.stdout.pipe.$$"
  START_STDERR_PIPE="${TMPDIR:-/tmp}/gaia-pipeline.stderr.pipe.$$"
  mkfifo "${START_STDOUT_PIPE}" "${START_STDERR_PIPE}"
  START_LOGGING=1
  tee "${RUNNER_STATE_DIR}/start.stdout" < "${START_STDOUT_PIPE}" &
  tee "${RUNNER_STATE_DIR}/start.stderr" >&2 < "${START_STDERR_PIPE}" &
  exec sh "$0" "$@" > "${START_STDOUT_PIPE}" 2> "${START_STDERR_PIPE}"
fi

VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
HF_HOME="${REPO_ROOT}/_state/huggingface"
PLAYWRIGHT_BROWSERS_PATH="${REPO_ROOT}/playwright-browsers"
PATH="${REPO_ROOT}/.venv/bin:${PATH}"

cd "${REPO_ROOT}"
mkdir -p \
  "${HF_HOME}" \
  "${INSPECT_LOG_DIR}" \
  "${PLAYWRIGHT_BROWSERS_PATH}" \
  "${RUNNER_STATE_DIR}" \
  "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"
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
rm -f "${XDG_RUNTIME_DIR}/inspect-tool-support.sock"
"${REPO_ROOT}/.venv/bin/inspect-tool-support" server \
  > "${RUNNER_STATE_DIR}/inspect-tool-support.log" 2>&1 & pid=$!
echo "${pid}" >> "${PID_FILE}"
i=0
while [ ! -S "${XDG_RUNTIME_DIR}/inspect-tool-support.sock" ]; do
  kill -0 "${pid}" 2>/dev/null || exit 1
  i=$((i+1)); [ "${i}" -le 30 ] || exit 1
  sleep 1
done

sh "${REPO_ROOT}/runner/base_model/${BASE_MODEL_RUNNER_TYPE}.sh"
sh "${REPO_ROOT}/runner/scaffold.sh"
sh "${REPO_ROOT}/runner/benchmark.sh"
