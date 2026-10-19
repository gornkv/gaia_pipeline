#!/bin/sh
set -aue

case "${GAIA_TASK}" in
  inspect_evals/gaia) GAIA_TASK_RUN="runner/gaia_task.py@gaia" ;;
  inspect_evals/gaia_level1) GAIA_TASK_RUN="runner/gaia_task.py@gaia_level1" ;;
  inspect_evals/gaia_level2) GAIA_TASK_RUN="runner/gaia_task.py@gaia_level2" ;;
  inspect_evals/gaia_level3) GAIA_TASK_RUN="runner/gaia_task.py@gaia_level3" ;;
  *) GAIA_TASK_RUN="${GAIA_TASK}" ;;
esac

set -- eval "${GAIA_TASK_RUN}" \
  --model "openai-api/gaia_model/${SCAFFOLD_MODEL_NAME}" \
  --sandbox "${GAIA_SANDBOX}" \
  --max-connections "${GAIA_MAX_CONNECTIONS}" \
  -T "split=${GAIA_SPLIT}" \
  -T "max_attempts=${GAIA_MAX_ATTEMPTS}" \
  -T "code_timeout=${GAIA_TOOL_TIMEOUT}" \
  --time-limit "${GAIA_SAMPLE_TIME_LIMIT}" \
  --message-limit "${GAIA_MESSAGE_LIMIT}" \
  --no-fail-on-error

if [ -n "${GAIA_SAMPLE_START:-}" ] && [ -n "${GAIA_SAMPLE_END:-}" ]; then
  set -- "$@" --limit "${GAIA_SAMPLE_START}-${GAIA_SAMPLE_END}"
fi

if [ -n "${GAIA_MAX_SAMPLES:-}" ]; then
  set -- "$@" --max-samples "${GAIA_MAX_SAMPLES}"
fi

"${REPO_ROOT}/.venv/bin/inspect" "$@"

LATEST="$(ls -t "${INSPECT_LOG_DIR}"/20*.eval 2>/dev/null | head -1)"
if [ -n "${LATEST}" ]; then
  RESULT_NAME="${BASE_MODEL_RUNNER_TYPE}_${FEATURE_NAME}${RUN_TAG:+_${RUN_TAG}}.eval"
  cp "${LATEST}" "${INSPECT_LOG_DIR}/${RESULT_NAME}"
  echo "Saved: ${INSPECT_LOG_DIR}/${RESULT_NAME}"
fi
