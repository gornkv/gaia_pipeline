#!/bin/sh
# Tear down the whole browser stack between rounds: tool-support servers, their
# playwright drivers, chromium, and the unix sockets.
#
# Never kill the driver alone: a tool-support server whose driver disappears stays
# alive but answers every browser call with
#   "Browser.new_context: Connection closed while reading from the driver"
# and the runner will happily reuse it (the socket file is still there).
#
# Refuses to run while any inspect eval is live, since that would kill its browsers.
ROOT=/home/jovyan/nachevsky/experiments/gaia_pipeline
ME=$$

live=0
for p in $(ls /proc 2>/dev/null | grep -E "^[0-9]+$"); do
  [ "$p" = "$ME" ] && continue
  c=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null)
  case "$c" in *clean_browsers*) continue ;; esac
  case "$c" in *"bin/inspect eval"*) live=$((live+1)) ;; esac
done
if [ "$live" -gt 0 ] && [ "$1" != "--force" ]; then
  echo "REFUSING: $live inspect eval process(es) still running. Harvest first, or pass --force."
  exit 1
fi

killed=0
for p in $(ls /proc 2>/dev/null | grep -E "^[0-9]+$"); do
  [ "$p" = "$ME" ] && continue
  c=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null)
  case "$c" in *clean_browsers*) continue ;; esac
  case "$c" in
    *inspect-tool-support*|*playwright*|*chromium*|*chrome-headless*|*chrome_crashpad*)
      kill -9 "$p" 2>/dev/null && killed=$((killed+1)) ;;
  esac
done

rm -f "$ROOT"/_state/run/inspect-tool-support.sock \
      "$ROOT"/_state/run_b/inspect-tool-support.sock \
      "$ROOT"/_state/run_a2/inspect-tool-support.sock \
      "$ROOT"/_state/run_b2/inspect-tool-support.sock

echo "killed=$killed processes; sockets removed. Runners will start fresh servers."
