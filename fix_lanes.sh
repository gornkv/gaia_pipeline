#!/bin/sh
# Kill evals + broken tool-support servers for the given lanes ONLY. Never touches other lanes.
# usage: sh fix_lanes.sh "19080 19081 19085" "run run_b run_b2"
PORTS="$1"; XDGS="$2"; ME=$$
for p in $(ls /proc | grep -E "^[0-9]+$"); do
  [ "$p" = "$ME" ] && continue
  c=$(tr "\0" " " </proc/$p/cmdline 2>/dev/null)
  case "$c" in *fix_lanes*) continue;; esac
  env=$(tr "\0" "\n" </proc/$p/environ 2>/dev/null)
  case "$c" in
    *"bin/inspect eval"*)
      for pt in $PORTS; do
        echo "$env" | grep -q "GAIA_MODEL_BASE_URL=http://127.0.0.1:$pt" && { echo "kill eval $p (port $pt)"; kill -9 $p; }
      done ;;
    *inspect-tool-support*)
      for x in $XDGS; do
        echo "$env" | grep -q "XDG_RUNTIME_DIR=.*/_state/$x$" && { echo "kill tool-support $p (xdg $x)"; kill -9 $p; }
      done ;;
  esac
done
for x in $XDGS; do
  rm -f "/home/jovyan/nachevsky/experiments/gaia_pipeline/_state/$x/inspect-tool-support.sock"
  echo "removed stale socket: _state/$x/inspect-tool-support.sock"
done
