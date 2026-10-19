#!/bin/sh
set -aue

"${VENV_PYTHON}" -m uvicorn svc_scaffold.main:app --ws none --host "0.0.0.0" --port "${SCAFFOLD_PORT}" \
  > "${RUNNER_STATE_DIR}/scaffold.stdout" \
  2> "${RUNNER_STATE_DIR}/scaffold.stderr" & pid=$!
echo "${pid}" >> "${PID_FILE}"

check_service() {
  "${VENV_PYTHON}" - <<'PY'
import os
from urllib.request import urlopen

try:
    urlopen(f"http://127.0.0.1:{os.environ['SCAFFOLD_PORT']}/health", timeout=5).close()
except Exception:
    raise SystemExit(1)
PY
}

while ! check_service; do
  kill -0 "$pid" 2>/dev/null || exit 1
  sleep 5
done
