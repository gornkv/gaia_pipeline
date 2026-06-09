#!/bin/sh
set -aue

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

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

mkdir -p \
  "${HF_HOME}" \
  "${INSPECT_LOG_DIR}" \
  "${PLAYWRIGHT_BROWSERS_PATH}" \
  "${REPO_ROOT}/_state/runner"
rm -rf "${REPO_ROOT}/_state/runner"/*

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
  git check-ref-format --branch "${LOGS_BRANCH}" >/dev/null

  base_ref="HEAD"
  if git show-ref --verify --quiet "refs/remotes/origin/${LOGS_BRANCH}"; then
    base_ref="origin/${LOGS_BRANCH}"
  elif git show-ref --verify --quiet "refs/heads/${LOGS_BRANCH}"; then
    base_ref="${LOGS_BRANCH}"
  fi

  logs_index="${REPO_ROOT}/_state/logs-index"
  rm -f "${logs_index}"

  GIT_INDEX_FILE="${logs_index}" git read-tree "${base_ref}"
  GIT_INDEX_FILE="${logs_index}" git add -f inspect-logs

  if GIT_INDEX_FILE="${logs_index}" git diff --cached --quiet -- inspect-logs; then
    echo "No inspect log changes to publish."
    rm -f "${logs_index}"
    return
  fi

  logs_tree="$(GIT_INDEX_FILE="${logs_index}" git write-tree)"
  logs_parent="$(git rev-parse "${base_ref}^{commit}")"
  logs_commit="$(git commit-tree "${logs_tree}" -p "${logs_parent}" -m "update inspect logs")"
  origin_url="$(git config --get remote.origin.url)"
  repo_path="${origin_url#git@github.com:}"
  repo_path="${repo_path#https://github.com/}"
  repo_path="${repo_path%.git}"
  push_url="https://github.com/${repo_path}.git"

  git update-ref "refs/heads/${LOGS_BRANCH}" "${logs_commit}"

  git -c credential.helper= \
    -c 'credential.helper=!f() {
        echo username=x-access-token
        echo password="$GITHUB_TOKEN"
    }; f' \
    push "${push_url}" "refs/heads/${LOGS_BRANCH}:refs/heads/${LOGS_BRANCH}"
  rm -f "${logs_index}"
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
