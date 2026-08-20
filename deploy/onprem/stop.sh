#!/usr/bin/env bash
# Stops the processes deploy/onprem/run.sh started, using the PIDs it saved under data/
# (or KEEL_DATA_DIR when set). A llama-server that was already running before run.sh has no pid file
# and is left alone.
#
# Flags: --keep-llama (stop the web app only)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${KEEL_DATA_DIR:-$REPO_ROOT/data}"

KEEP_LLAMA=0
for arg in "$@"; do
  case "$arg" in
    --keep-llama) KEEP_LLAMA=1 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "stop.sh: unknown flag $arg (known: --keep-llama)"; exit 2 ;;
  esac
done

stop_from_pid_file() {
  # stop_from_pid_file NAME
  local name="$1" pid_file="$DATA_DIR/$1.pid" pid
  if [ ! -f "$pid_file" ]; then
    echo "$name: no pid file at $pid_file, nothing to stop."
    return
  fi
  pid="$(head -n 1 "$pid_file" | tr -d '[:space:]')"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if ! kill -0 "$pid" 2>/dev/null; then break; fi
      sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid" 2>/dev/null || true; fi
    echo "$name: pid $pid stopped."
  else
    echo "$name: pid $pid already gone."
  fi
  rm -f "$pid_file"
}

stop_from_pid_file "keel-web"
if [ "$KEEP_LLAMA" = 1 ]; then
  echo "llama-server: kept running (--keep-llama)."
else
  stop_from_pid_file "llama-server"
fi
