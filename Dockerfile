# Keel app image. Serves the web UI and API (8400 by default, or Railway's PORT) and talks to a
# llama.cpp server named by KEEL_LOCAL_LLM_BASE_URL. deploy/onprem/docker-compose.yml wires it to the
# `llama` service; deploy/railway/README.md wires it to the `keel-llm` service on Railway.
#
#   docker build -t keel .
#   docker run --rm -p 127.0.0.1:8400:8400 -v keel-data:/data keel
#
# Embedding models: fastembed reads FASTEMBED_CACHE_PATH. The build prefetches the two default models
# (bge-small-en-v1.5, ms-marco-MiniLM-L-6-v2) when PREFETCH_MODELS=1 (default) so the image works
# with HF_HUB_OFFLINE=1 and the first request is fast. For a build host with no network, copy an
# existing cache into deploy/onprem/model-cache/ first (on Windows it lives at %TEMP%\fastembed_cache,
# on Linux at /tmp/fastembed_cache) and build with --build-arg PREFETCH_MODELS=0; the directory is
# copied into the image either way.
#
# The server runs as the unprivileged user `keel` (uid 10001): the entrypoint starts as root, makes
# KEEL_DATA_DIR=/data (a mounted volume, which platforms mount root-owned; Railway also rejects a
# Dockerfile VOLUME instruction) writable by keel, and drops privileges with setpriv before uvicorn
# starts. Runtime env: see deploy/railway/README.md.

FROM python:3.11-slim

ARG PREFETCH_MODELS=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    FASTEMBED_CACHE_PATH=/home/keel/.cache/fastembed \
    KEEL_DATA_DIR=/data \
    KEEL_HOST=0.0.0.0

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin keel

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY keel ./keel
COPY fixtures ./fixtures
COPY scripts ./scripts
COPY deploy/railway/entrypoint.sh /usr/local/bin/keel-entrypoint

# Editable install keeps the web templates and static files served from /app/keel.
RUN pip install --no-cache-dir -e . \
    && chmod 0755 /usr/local/bin/keel-entrypoint \
    && mkdir -p /data /home/keel/.cache/fastembed \
    && chown -R keel:keel /app /data /home/keel

# Bake a pre-existing fastembed cache when the build host has one staged (see the note above).
COPY --chown=keel:keel deploy/onprem/model-cache/ /home/keel/.cache/fastembed/

USER keel

RUN if [ "$PREFETCH_MODELS" = "1" ]; then \
      python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2'); print('models cached')"; \
    else echo "PREFETCH_MODELS=0: relying on deploy/onprem/model-cache"; fi

# The entrypoint starts as root only to chown the data volume, then drops to `keel` (see
# deploy/railway/entrypoint.sh). Everything above ran as keel so the model cache is theirs.
USER root

EXPOSE 8400

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8400'), timeout=4)"

ENTRYPOINT ["/usr/local/bin/keel-entrypoint"]
