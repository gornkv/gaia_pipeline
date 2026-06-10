#!/bin/sh
set -aue

pkill -f '[u]vicorn svc_scaffold.main:app' 2>/dev/null && sleep 2

"${VENV_PYTHON}" -m uvicorn svc_scaffold.main:app --host "0.0.0.0" --port "19080" \
  > "${REPO_ROOT}/_state/runner/scaffold.stdout" \
  2> "${REPO_ROOT}/_state/runner/scaffold.stderr" & pid=$!
echo "${pid}" >> "${PID_FILE}"

check_service() {
  "${VENV_PYTHON}" - <<'PY'
from urllib.request import urlopen

try:
    urlopen("http://127.0.0.1:19080/health", timeout=5).close()
except Exception:
    raise SystemExit(1)
PY
}

while ! check_service; do
  kill -0 "$pid" 2>/dev/null || exit 1
  sleep 5
done
