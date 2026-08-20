#!/bin/sh
# Keel app entrypoint. Railway sets PORT; every other host gets 8400. KEEL_HOST defaults to 0.0.0.0
# inside the container (the image sits behind Railway's edge or a reverse proxy) and the admin
# guard, which keys off KEEL_HOST, is therefore on: /admin needs X-Keel-Admin-Token.
#
# The container starts as root only long enough to make the data directory writable by `keel`
# (a platform volume mounts root-owned), then drops to uid 10001 with setpriv. Started as any other
# user (docker run --user), it skips the chown and runs as that user.
set -eu
PORT="${PORT:-8400}"
export KEEL_HOST="${KEEL_HOST:-0.0.0.0}"
export KEEL_PORT="$PORT"
DATA_DIR="${KEEL_DATA_DIR:-/data}"
mkdir -p "$DATA_DIR"

if [ "$(id -u)" = "0" ]; then
  chown -R keel:keel "$DATA_DIR"
  exec setpriv --reuid=keel --regid=keel --init-groups \
    python -m uvicorn keel.web.app:app --host "$KEEL_HOST" --port "$PORT" --proxy-headers --forwarded-allow-ips '*'
fi
exec python -m uvicorn keel.web.app:app --host "$KEEL_HOST" --port "$PORT" --proxy-headers --forwarded-allow-ips '*'
