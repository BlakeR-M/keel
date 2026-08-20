#!/usr/bin/env bash
# Starts the Keel on-premise stack without Docker: llama-server on 127.0.0.1:8081, then the Keel web app
# on 127.0.0.1:8400. Idempotent: a service already answering is left alone. Background processes record
# their PIDs under data/ so stop.sh can end them; output goes to data/llama-server.{out,err} and
# data/keel-web.{out,err}.
#
# Environment (every value has a default):
#   KEEL_LLAMA_SERVER      llama-server binary          default: llama-server on PATH
#                          (falls back to /d/llama.cpp/bin/llama-server.exe under Git Bash on Windows)
#   KEEL_MODEL_PATH        GGUF model file              default: ./models/qwen2.5-3b-instruct-q4_k_m.gguf,
#                          or /d/models/... and /mnt/d/models/... when one of those exists
#   KEEL_NGL               GPU layers to offload        default 0 (CPU only)
#   KEEL_LLAMA_CTX         context length               default 8192
#   KEEL_LLAMA_THREADS     CPU threads                  default nproc
#   KEEL_LLAMA_ALIAS       model alias served           default qwen2.5-3b-instruct
#   KEEL_LLAMA_PORT        llama-server port            default 8081
#   KEEL_LLAMA_EXTRA_ARGS  extra llama-server flags     default none
#   KEEL_PYTHON            interpreter for the web app  default .venv/bin/python (or .venv/Scripts/python.exe)
#   KEEL_WEB_APP           ASGI app for uvicorn         default keel.web.app:app
#   KEEL_DATA_DIR          pid and log directory        default <repo>/data
#   KEEL_AIRGAP            1 turns on the egress guard in the app (passed through unchanged)
#
# Flags: --skip-web (llama-server only), --skip-llama (web app only), --timeout SECONDS (default 300)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SKIP_WEB=0
SKIP_LLAMA=0
TIMEOUT=300
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-web) SKIP_WEB=1 ;;
    --skip-llama) SKIP_LLAMA=1 ;;
    --timeout) TIMEOUT="$2"; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "run.sh: unknown flag $1 (known: --skip-web, --skip-llama, --timeout SECONDS)"; exit 2 ;;
  esac
  shift
done

DATA_DIR="${KEEL_DATA_DIR:-$REPO_ROOT/data}"
mkdir -p "$DATA_DIR"

# --- defaults -----------------------------------------------------------------------------------
default_model() {
  for candidate in "$REPO_ROOT/models/qwen2.5-3b-instruct-q4_k_m.gguf" \
                   /d/models/qwen2.5-3b-instruct-q4_k_m.gguf \
                   /mnt/d/models/qwen2.5-3b-instruct-q4_k_m.gguf; do
    if [ -f "$candidate" ]; then echo "$candidate"; return; fi
  done
  echo "$REPO_ROOT/models/qwen2.5-3b-instruct-q4_k_m.gguf"
}

default_llama_server() {
  if command -v llama-server >/dev/null 2>&1; then echo "llama-server"; return; fi
  if [ -x /d/llama.cpp/bin/llama-server.exe ]; then echo "/d/llama.cpp/bin/llama-server.exe"; return; fi
  echo "llama-server"
}

default_python() {
  if [ -x "$REPO_ROOT/.venv/bin/python" ]; then echo "$REPO_ROOT/.venv/bin/python"; return; fi
  if [ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]; then echo "$REPO_ROOT/.venv/Scripts/python.exe"; return; fi
  echo "python3"
}

LLAMA_SERVER="${KEEL_LLAMA_SERVER:-$(default_llama_server)}"
MODEL_PATH="${KEEL_MODEL_PATH:-$(default_model)}"
NGL="${KEEL_NGL:-0}"
CTX="${KEEL_LLAMA_CTX:-8192}"
THREADS="${KEEL_LLAMA_THREADS:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8) )}"
ALIAS="${KEEL_LLAMA_ALIAS:-qwen2.5-3b-instruct}"
LLAMA_PORT="${KEEL_LLAMA_PORT:-8081}"
EXTRA_ARGS="${KEEL_LLAMA_EXTRA_ARGS:-}"
PY="${KEEL_PYTHON:-$(default_python)}"
WEB_APP="${KEEL_WEB_APP:-keel.web.app:app}"
WEB_PORT=8400

LLAMA_MODELS_URL="http://127.0.0.1:$LLAMA_PORT/v1/models"
WEB_URL="http://127.0.0.1:$WEB_PORT/"

# --- helpers ------------------------------------------------------------------------------------
http_ok() {
  # 2xx answer. llama-server answers 503 while the model is still loading.
  "$PY" - "$1" <<'PYEOF' 2>/dev/null
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=3) as r:
        sys.exit(0 if 200 <= r.status < 300 else 1)
except Exception:
    sys.exit(1)
PYEOF
}

http_answers() {
  # Any HTTP answer, status code aside: the server is up.
  "$PY" - "$1" <<'PYEOF' 2>/dev/null
import sys, urllib.request, urllib.error
try:
    urllib.request.urlopen(sys.argv[1], timeout=3)
    sys.exit(0)
except urllib.error.HTTPError:
    sys.exit(0)
except Exception:
    sys.exit(1)
PYEOF
}

wait_for() {
  # wait_for NAME CHECK_FN PID SECONDS ERR_LOG
  local name="$1" check="$2" pid="$3" seconds="$4" err_log="$5"
  local deadline=$(( $(date +%s) + seconds ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if "$check"; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited before it was ready. Last lines of $err_log:"
      tail -n 15 "$err_log" 2>/dev/null || true
      return 1
    fi
    sleep 0.5
  done
  echo "$name did not answer within $seconds seconds. Check $err_log, or pass --timeout for a large model."
  return 1
}

check_llama() { http_ok "$LLAMA_MODELS_URL"; }
check_web() { http_answers "$WEB_URL"; }

# --- llama-server -------------------------------------------------------------------------------
if [ "$SKIP_LLAMA" = 1 ]; then
  echo "llama-server: skipped (--skip-llama). Expecting an OpenAI-compatible endpoint at $LLAMA_MODELS_URL"
elif check_llama; then
  echo "llama-server: already answering at $LLAMA_MODELS_URL, leaving it running."
else
  if ! command -v "$LLAMA_SERVER" >/dev/null 2>&1 && [ ! -x "$LLAMA_SERVER" ]; then
    echo "llama-server binary missing ($LLAMA_SERVER). Install llama.cpp and put llama-server on PATH, or set KEEL_LLAMA_SERVER."
    exit 1
  fi
  if [ ! -f "$MODEL_PATH" ]; then
    echo "Model file missing at $MODEL_PATH. Set KEEL_MODEL_PATH to a GGUF file (see docs/onprem.md for the model swap)."
    exit 1
  fi
  # shellcheck disable=SC2086
  set -- -m "$MODEL_PATH" --host 127.0.0.1 --port "$LLAMA_PORT" --jinja --alias "$ALIAS" -ngl "$NGL" -c "$CTX" -t "$THREADS" $EXTRA_ARGS
  echo "llama-server: starting $LLAMA_SERVER $*"
  nohup "$LLAMA_SERVER" "$@" >"$DATA_DIR/llama-server.out" 2>"$DATA_DIR/llama-server.err" &
  LLAMA_PID=$!
  echo "$LLAMA_PID" >"$DATA_DIR/llama-server.pid"
  echo "llama-server: pid $LLAMA_PID, waiting for $LLAMA_MODELS_URL (model load can take a minute on CPU)"
  wait_for "llama-server" check_llama "$LLAMA_PID" "$TIMEOUT" "$DATA_DIR/llama-server.err"
  echo "llama-server: ready at http://127.0.0.1:$LLAMA_PORT/v1 (alias $ALIAS, ngl $NGL)"
fi

# --- Keel web app -------------------------------------------------------------------------------
if [ "$SKIP_WEB" = 1 ]; then
  echo "keel web: skipped (--skip-web)."
  exit 0
fi
if check_web; then
  echo "keel web: already answering at $WEB_URL, leaving it running."
  exit 0
fi
if ! command -v "$PY" >/dev/null 2>&1 && [ ! -x "$PY" ]; then
  echo "Python interpreter missing ($PY). Run 'make demo' once (it creates .venv), or set KEEL_PYTHON."
  exit 1
fi
export KEEL_LOCAL_LLM_BASE_URL="${KEEL_LOCAL_LLM_BASE_URL:-http://127.0.0.1:$LLAMA_PORT/v1}"
export KEEL_LOCAL_LLM_MODEL="${KEEL_LOCAL_LLM_MODEL:-$ALIAS}"
export HF_HUB_DISABLE_SYMLINKS_WARNING="${HF_HUB_DISABLE_SYMLINKS_WARNING:-1}"
export KEEL_DATA_DIR="$DATA_DIR"

echo "keel web: starting $PY -m uvicorn $WEB_APP --host 127.0.0.1 --port $WEB_PORT"
cd "$REPO_ROOT"
nohup "$PY" -m uvicorn "$WEB_APP" --host 127.0.0.1 --port "$WEB_PORT" \
  >"$DATA_DIR/keel-web.out" 2>"$DATA_DIR/keel-web.err" &
WEB_PID=$!
echo "$WEB_PID" >"$DATA_DIR/keel-web.pid"
wait_for "keel web" check_web "$WEB_PID" "$TIMEOUT" "$DATA_DIR/keel-web.err"
echo "keel web: ready at $WEB_URL (pid $WEB_PID). Stop everything with deploy/onprem/stop.sh"
