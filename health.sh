#!/bin/sh
# gpu_mem load chromium threads  lanes_with_live_browser  live_evals
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk "{s+=\$1} END{print s+0}")
la=$(cut -d" " -f2 /proc/loadavg | cut -d. -f1)
ch=$(ps -eo stat,comm | grep -i chrom | grep -vc "^Z")
th=$(cut -d" " -f4 /proc/loadavg)
lanes=""; ev=0
for p in $(ls /proc 2>/dev/null | grep -E "^[0-9]+$"); do
  c=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null)
  case "$c" in
    *"bin/inspect eval"*) ev=$((ev+1)) ;;
    *inspect-tool-support*)
      x=$(tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep "^XDG_RUNTIME_DIR=" | sed "s#.*/_state/##")
      case " $lanes " in *" $x "*) ;; *) lanes="$lanes $x" ;; esac ;;
  esac
done
nl=$(echo $lanes | wc -w)
echo "$mem $la $ch $th $nl $ev"
