#!/bin/sh
# Fetch the GGUF into /models (a volume) when it is absent, then run llama-server on all interfaces.
# Railway sets PORT for services with a public domain; this one is private, so LLAMA_PORT (8081) is
# the port unless PORT is given.
set -eu
MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_PATH="$MODEL_DIR/${MODEL_FILE:?MODEL_FILE is required}"
PORT="${PORT:-${LLAMA_PORT:-8081}}"
THREADS="${LLAMA_THREADS:-$(nproc)}"

mkdir -p "$MODEL_DIR"
if [ ! -s "$MODEL_PATH" ]; then
  echo "keel-llm: downloading $MODEL_URL to $MODEL_PATH"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$MODEL_PATH.part" "$MODEL_URL"
  mv "$MODEL_PATH.part" "$MODEL_PATH"
  echo "keel-llm: download complete ($(du -h "$MODEL_PATH" | cut -f1))"
else
  echo "keel-llm: using cached model $MODEL_PATH ($(du -h "$MODEL_PATH" | cut -f1))"
fi

echo "keel-llm: starting llama-server on 0.0.0.0:$PORT with $THREADS threads"
exec /app/llama-server \
  -m "$MODEL_PATH" \
  --host 0.0.0.0 --port "$PORT" \
  --jinja \
  --alias "${MODEL_ALIAS:-qwen2.5-3b-instruct}" \
  -c "${LLAMA_CTX:-8192}" \
  -t "$THREADS" \
  --parallel "${LLAMA_PARALLEL:-2}" \
  ${LLAMA_EXTRA_ARGS:-}
