#!/bin/sh
# Kill runner shells matching $1, skipping this script own process tree.
PAT="$1"; ME=$$
for p in $(ls /proc 2>/dev/null | grep -E "^[0-9]+$"); do
  [ "$p" = "$ME" ] && continue
  c=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null)
  case "$c" in *stop_runners*) continue ;; esac
  case "$c" in *"$PAT"*) kill -9 "$p" 2>/dev/null && echo "killed $p" ;; esac
done
