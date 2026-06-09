#!/bin/sh
set -aue

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "${REPO_ROOT}/_state/runner"
if [ "${START_LOGGING:-}" != "1" ]; then
  rm -rf "${REPO_ROOT}/_state/runner"/*
  START_STDOUT_PIPE="${REPO_ROOT}/_state/runner/start.stdout.pipe.$$"
  START_STDERR_PIPE="${REPO_ROOT}/_state/runner/start.stderr.pipe.$$"
  mkfifo "${START_STDOUT_PIPE}" "${START_STDERR_PIPE}"
  START_LOGGING=1
  tee "${REPO_ROOT}/_state/runner/start.stdout" < "${START_STDOUT_PIPE}" &
  tee "${REPO_ROOT}/_state/runner/start.stderr" >&2 < "${START_STDERR_PIPE}" &
  exec sh "$0" "$@" > "${START_STDOUT_PIPE}" 2> "${START_STDERR_PIPE}"
fi

load_env() {
  mkdir -p "${REPO_ROOT}/_state/env"
  tr -d '\r' < "$1" > "${REPO_ROOT}/_state/env/$(basename "$1")"
  . "${REPO_ROOT}/_state/env/$(basename "$1")"
}

load_env "${REPO_ROOT}/.env"
cd "${REPO_ROOT}"

VENV_PYTHON="${REPO_ROOT}/_state/.venv/bin/python"
HF_HOME="${REPO_ROOT}/_state/huggingface"
INSPECT_LOG_DIR="${REPO_ROOT}/inspect-logs"
PLAYWRIGHT_BROWSERS_PATH="${REPO_ROOT}/_state/playwright-browsers"
PATH="${REPO_ROOT}/_state/.venv/bin:${PATH}"
PID_FILE="${REPO_ROOT}/_state/runner/pids"

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

  if "${PYTHON_BIN:-python3}" -m venv "${REPO_ROOT}/_state/.venv"; then
    return
  fi

  rm -rf "${REPO_ROOT}/_state/.venv"
  "${PYTHON_BIN:-python3}" -m venv --system-site-packages --without-pip "${REPO_ROOT}/_state/.venv"
  "${VENV_PYTHON}" -m pip install --upgrade pip wheel setuptools
}

install_environment() {
  create_venv
  "${VENV_PYTHON}" -m pip install \
    pip wheel setuptools "inspect-evals[gaia]" inspect-tool-support openai playwright \
    -r "${REPO_ROOT}/svc_scaffold/requirement.txt"
  "${REPO_ROOT}/_state/.venv/bin/inspect-tool-support" post-install
  "${VENV_PYTHON}" -m playwright install chromium
  "${VENV_PYTHON}" "${REPO_ROOT}/runner/check_environment.py"
}

publish_inspect_logs() {
  [ -n "${LOGS_BRANCH:-}" ] || return
  git fetch origin
  git switch -C "${LOGS_BRANCH}" "origin/${LOGS_BRANCH}" 2>/dev/null ||
    git switch -C "${LOGS_BRANCH}"
  git add inspect-logs
  git -c user.name="gaia-pipeline" -c user.email="gaia-pipeline@example.invalid" commit -m "update inspect logs"
  git -c credential.helper= \
    -c 'credential.helper=!f() {
      echo username=x-access-token
      echo password="$GITHUB_TOKEN"
    }; f' push origin "${LOGS_BRANCH}"
}

if ! "${VENV_PYTHON}" "${REPO_ROOT}/runner/check_environment.py" >/dev/null 2>&1; then
  install_environment
fi

RUN_TASK_NAME="2_run_base_model_${BASE_MODEL_RUNNER_TYPE}"
load_env "${REPO_ROOT}/runner/${RUN_TASK_NAME}.env"
sh "${REPO_ROOT}/runner/${RUN_TASK_NAME}.sh"

load_env "${REPO_ROOT}/runner/3_run_scaffold.env"
sh "${REPO_ROOT}/runner/3_run_scaffold.sh"

sh "${REPO_ROOT}/runner/4_run_benchmark.sh"

publish_inspect_logs
